import cv2
import mediapipe as mp
import math
import time
import streamlit as st

# ==== Setup ====
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

# ==== Streamlit UI ====
st.title("👁 Blink & Distance Monitor")
st.write("Metrics only (no video stream).")

blink_box = st.empty()
ear_box = st.empty()
dist_box = st.empty()

EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3
MIN_BLINKS_PER_MINUTE = 15

COUNTER = 0
TOTAL = 0
LIFETIME = 0
last_reset = time.time()

cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
        if TOTAL < MIN_BLINKS_PER_MINUTE:
            st.warning(f"⚠️ Low blink rate: {TOTAL} in last minute")
        TOTAL = 0
        last_reset = time.time()

    # ==== Update metrics live ====
    blink_box.metric("Blinks (last min)", TOTAL, help=f"Lifetime: {LIFETIME}")
    ear_box.metric("EAR", f"{ear:.2f}" if ear else "–")
    dist_box.metric("Distance (cm)", f"{iris_dist_cm:.1f}" if iris_dist_cm else "–")

    # Tiny delay so UI updates properly
    time.sleep(0.05)

cap.release()
