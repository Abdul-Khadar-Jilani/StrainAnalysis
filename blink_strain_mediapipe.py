# filename: blink_strain_mediapipe.py
import argparse
import time
import threading
import math
import numpy as np
import cv2

# --- optional popup (Windows-friendly) ---
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    tk = None
    messagebox = None

# --- mediapipe ---
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

# ===== Eye landmark indices (MediaPipe Face Mesh, with refine_landmarks=True) =====
# We'll use 6 points per eye analogous to dlib's EAR calculation.
# Left eye: 33 (outer), 133 (inner), vertical pairs around eyelids.
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]   # [p0, p1, p2, p3, p4, p5]  -> (1,5) & (2,4) vertical; (0,3) horizontal
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Iris landmarks in Face Mesh with refine_landmarks=True
LEFT_IRIS_IDX = [468, 469, 470, 471]   # 4 iris points
RIGHT_IRIS_IDX = [473, 474, 475, 476]

IRIS_DIAMETER_MM = 11.7   # average human iris size in millimeters
FOCAL_LENGTH_PX = 600    # you can calibrate this for your webcam

def iris_distance(landmarks, iris_idx, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in iris_idx]
    # use horizontal distance between leftmost and rightmost iris points
    #iris_d_px = np.linalg.norm(np.array(pts[0]) - np.array(pts[2]))
    xs = [p[0] for p in pts]
    iris_d_px = max(xs) - min(xs)   # horizontal iris diameter

    #print(iris_d_px)
    # for i in LEFT_IRIS_IDX:
    #     lm = landmarks[i]
    #     print(i, lm.x * w, lm.y * h)

    if iris_d_px == 0:  
        return None
    # apply pinhole camera model
    dist_mm = (FOCAL_LENGTH_PX * IRIS_DIAMETER_MM) / iris_d_px
    return dist_mm / 10.0  # convert mm -> cm

def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def eye_aspect_ratio(eye_pts):
    """
    eye_pts: list of 6 (x,y) tuples in this order:
      [p0, p1, p2, p3, p4, p5]
    EAR = (||p1 - p5|| + ||p2 - p4||) / (2 * ||p0 - p3||)
    """
    A = euclidean(eye_pts[1], eye_pts[5])
    B = euclidean(eye_pts[2], eye_pts[4])
    C = euclidean(eye_pts[0], eye_pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def safe_popup(msg):
    """Show a non-blocking popup (only if tkinter available)."""
    if tk is None or messagebox is None:
        print(f"[ALERT] {msg}")
        return

    def _show():
        root = tk.Tk()
        root.withdraw()
        try:
            messagebox.showwarning("Eye Strain Alert", msg, parent=root)
        finally:
            root.destroy()
    t = threading.Thread(target=_show, daemon=True)
    t.start()

def draw_eye_contour(frame, pts):
    """Draw simple polyline around eye points."""
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [pts_arr], isClosed=True, thickness=1, color=(0, 255, 0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, default="",
                    help="path to input video file (default: webcam)")
    ap.add_argument("--ear-thresh", type=float, default=0.27,
                    help="EAR threshold for blink detection (default: 0.27)")
    ap.add_argument("--consec-frames", type=int, default=3,
                    help="Consecutive frames below thresh to count a blink (default: 3)")
    ap.add_argument("--min-blinks-per-min", type=int, default=15,
                    help="Minimum healthy blinks per minute before alert (default: 15)")
    args = ap.parse_args()

    EYE_AR_THRESH = args.ear_thresh
    EYE_AR_CONSEC_FRAMES = args.consec_frames
    MIN_BLINKS_PER_MINUTE = args.min_blinks_per_min

    COUNTER = 0  # frames below threshold
    TOTAL = 0    # total blinks in current minute window
    last_reset = time.time()

    # Video source
    if args.video.strip() == "":
        print("[INFO] starting webcam...")
        cap = cv2.VideoCapture(0)
    else:
        print("[INFO] opening video file...")
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    # MediaPipe Face Mesh (refine_landmarks=True gives better eye/iris points)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
            h, w = frame.shape[:2]

            # BGR -> RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # If a minute passed, evaluate blink rate
            elapsed = time.time() - last_reset
            if elapsed >= 60.0:
                # Simple check: TOTAL is blinks in the past minute
                if TOTAL < MIN_BLINKS_PER_MINUTE:
                    msg = f"Low blink rate detected: {TOTAL} blinks in last minute. Please rest your eyes!"
                    safe_popup(msg)
                    print("[ALERT]", msg)
                TOTAL = 0
                last_reset = time.time()

            ear = None
            if results.multi_face_landmarks:
                # Use first detected face
                face_landmarks = results.multi_face_landmarks[0].landmark

                # Collect eye points (x,y) in pixel coords
                def idx_pts(idx_list):
                    pts = []
                    for i in idx_list:
                        lm = face_landmarks[i]
                        pts.append((int(lm.x * w), int(lm.y * h)))
                    return pts

                left_eye_pts = idx_pts(LEFT_EYE_IDX)
                right_eye_pts = idx_pts(RIGHT_EYE_IDX)

                # EAR per eye and average
                leftEAR = eye_aspect_ratio(left_eye_pts)
                rightEAR = eye_aspect_ratio(right_eye_pts)
                ear = (leftEAR + rightEAR) / 2.0

                # Draw contours
                draw_eye_contour(frame, left_eye_pts)
                draw_eye_contour(frame, right_eye_pts)

                # Blink logic
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                    COUNTER = 0
                # Iris distance (approx. eye-to-screen distance)
                iris_dist_cm = iris_distance(face_landmarks, LEFT_IRIS_IDX, w, h)
                if iris_dist_cm:
                    cv2.putText(frame, f"Distance: {iris_dist_cm:.1f} cm", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if iris_dist_cm < 20:
                        cv2.putText(frame, "Too Close!", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif iris_dist_cm > 80:
                        cv2.putText(frame, "Too Far!", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            # HUD
            cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if ear is not None:
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Strain Analyser (MediaPipe)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
