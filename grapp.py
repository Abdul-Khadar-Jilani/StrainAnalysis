# import cv2
# import mediapipe as mp
# import numpy as np
# import math
# import time
# import gradio as gr

# # ===== Setup Mediapipe =====
# mp_face_mesh = mp.solutions.face_mesh
# LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# LEFT_IRIS_IDX = [468, 469, 470, 471]

# IRIS_DIAMETER_MM = 11.7
# FOCAL_LENGTH_PX = 600  # calibrated for your cam

# # ===== Blink & Distance logic =====
# def euclidean(p1, p2):
#     return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# def eye_aspect_ratio(eye_pts):
#     A = euclidean(eye_pts[1], eye_pts[5])
#     B = euclidean(eye_pts[2], eye_pts[4])
#     C = euclidean(eye_pts[0], eye_pts[3])
#     return (A + B) / (2.0 * C) if C != 0 else 0

# def iris_distance(landmarks, iris_idx, w, h):
#     pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in iris_idx]
#     xs = [p[0] for p in pts]
#     iris_d_px = max(xs) - min(xs)
#     if iris_d_px == 0:
#         return None
#     dist_mm = (FOCAL_LENGTH_PX * IRIS_DIAMETER_MM) / iris_d_px
#     return dist_mm / 10.0  # cm

# # ===== State variables =====
# COUNTER = 0
# TOTAL = 0
# LIFETIME = 0
# last_reset = time.time()
# EYE_AR_THRESH = 0.27
# EYE_AR_CONSEC_FRAMES = 3
# MIN_BLINKS_PER_MINUTE = 15

# # ===== Main processing function =====
# def process_frame(frame):
#     global COUNTER, TOTAL, LIFETIME, last_reset
#     # Make frame writable for OpenCV
#     frame = frame.copy()
#     h, w, _ = frame.shape
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     with mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as face_mesh:
#     # Create FaceMesh once, globally
#     # face_mesh = mp_face_mesh.FaceMesh(
#     #     max_num_faces=1,
#     #     refine_landmarks=True,
#     #     min_detection_confidence=0.5,
#     #     min_tracking_confidence=0.5)
#         results = face_mesh.process(rgb)

#     ear, iris_dist_cm = None, None

#     if results.multi_face_landmarks:
#         face_landmarks = results.multi_face_landmarks[0].landmark

#         def idx_pts(idx_list):
#             return [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in idx_list]

#         left_eye_pts = idx_pts(LEFT_EYE_IDX)
#         right_eye_pts = idx_pts(RIGHT_EYE_IDX)

#         # EAR
#         leftEAR = eye_aspect_ratio(left_eye_pts)
#         rightEAR = eye_aspect_ratio(right_eye_pts)
#         ear = (leftEAR + rightEAR) / 2.0

#         # Blink detection
#         if ear < EYE_AR_THRESH:
#             COUNTER += 1
#         else:
#             if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                 TOTAL += 1
#                 LIFETIME += 1
#             COUNTER = 0

#         # Distance
#         iris_dist_cm = iris_distance(face_landmarks, LEFT_IRIS_IDX, w, h)

#         # Draw overlay
#         cv2.polylines(frame, [np.array(left_eye_pts, dtype=np.int32)], True, (0,255,0), 1)
#         cv2.polylines(frame, [np.array(right_eye_pts, dtype=np.int32)], True, (0,255,0), 1)
#         cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
#         if ear:
#             cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
#         if iris_dist_cm:
#             cv2.putText(frame, f"Dist: {iris_dist_cm:.1f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

#     # Reset blink count every 60s
#     elapsed = time.time() - last_reset
#     if elapsed >= 60:
#         if TOTAL < MIN_BLINKS_PER_MINUTE:
#             print(f"⚠️ Low blink rate: {TOTAL} blinks in last minute")
#         TOTAL = 0
#         last_reset = time.time()

#     # Text output
#     blink_text = f"Blinks (last min): {TOTAL} | Lifetime: {LIFETIME}"
#     ear_text = f"EAR: {ear:.2f}" if ear else "EAR: –"
#     dist_text = f"Distance: {iris_dist_cm:.1f} cm" if iris_dist_cm else "Distance: –"

#     #return frame, 
#     return blink_text, ear_text, dist_text

# # ===== Gradio UI =====
# demo = gr.Interface(
#     fn=process_frame,
#     inputs=gr.Image(sources="webcam", streaming=True, type="numpy"),
#     outputs=[#gr.Image(type="numpy"), 
#              gr.Textbox(), gr.Textbox(), gr.Textbox()],
#     live=True,
#     title="👁 Eye Strain & Distance Monitor",
#     description="Tracks blinks and eye-to-camera distance in real-time."
# )

# if __name__ == "__main__":
#     demo.launch()

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import gradio as gr

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX = [468, 469, 470, 471]
IRIS_DIAMETER_MM = 11.7
FOCAL_LENGTH_PX = 600

def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def eye_aspect_ratio(eye_pts):
    A = euclidean(eye_pts[1], eye_pts[5])
    B = euclidean(eye_pts[2], eye_pts[4])
    C = euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def iris_distance(landmarks, iris_idx, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in iris_idx]
    xs = [p[0] for p in pts]
    iris_d_px = max(xs) - min(xs)
    if iris_d_px == 0:
        return None
    dist_mm = (FOCAL_LENGTH_PX * IRIS_DIAMETER_MM) / iris_d_px
    return dist_mm / 10.0  # cm

# load FaceMesh once
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3
MIN_BLINKS_PER_MINUTE = 15

def process_frame(frame, state):
    if state is None:
        state = [0, 0, 0, time.time()]  # COUNTER, TOTAL, LIFETIME, last_reset

    COUNTER, TOTAL, LIFETIME, last_reset = state

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    ear, iris_dist_cm = None, None
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        def idx_pts(idx_list):
            return [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in idx_list]

        left_eye_pts = idx_pts(LEFT_EYE_IDX)
        right_eye_pts = idx_pts(RIGHT_EYE_IDX)

        leftEAR = eye_aspect_ratio(left_eye_pts)
        rightEAR = eye_aspect_ratio(right_eye_pts)
        ear = (leftEAR + rightEAR) / 2.0

        # Blink detection
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
                LIFETIME += 1
            COUNTER = 0

        iris_dist_cm = iris_distance(face_landmarks, LEFT_IRIS_IDX, w, h)

    elapsed = time.time() - last_reset
    if elapsed >= 60:
        TOTAL = 0
        last_reset = time.time()

    # --- Debug printouts ---
    debug_text = f"EAR: {ear:.2f} | COUNTER: {COUNTER} | TOTAL: {TOTAL}"

    blink_text = f"Blinks (last min): {TOTAL} | Lifetime: {LIFETIME}"
    ear_text = f"EAR: {ear:.2f}" if ear else "EAR: –"
    dist_text = f"Distance: {iris_dist_cm:.1f} cm" if iris_dist_cm else "Distance: –"

    return blink_text, ear_text, dist_text, debug_text, [COUNTER, TOTAL, LIFETIME, last_reset]

demo = gr.Interface(
    fn=process_frame,
    inputs=[gr.Image(sources="webcam", streaming=True, type="numpy"), gr.State()],
    outputs=[
        gr.Textbox(label="Blinks"),
        gr.Textbox(label="EAR"),
        gr.Textbox(label="Distance"),
        gr.Textbox(label="Debug (EAR | COUNTER | TOTAL)"),
        gr.State()
    ],
    live=True,
)



if __name__ == "__main__":
    demo.launch()
