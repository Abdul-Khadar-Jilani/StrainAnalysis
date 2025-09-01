import cv2
import mediapipe as mp
import numpy as np
import math
import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ==== Config ====
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX = [468, 469, 470, 471]

IRIS_DIAMETER_MM = 11.7
FOCAL_LENGTH_PX = 600
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3
MIN_BLINKS_PER_MINUTE = 15

# ==== Utils ====
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
st.set_page_config(page_title="Eye Strain Monitor", page_icon="👁", layout="wide")
st.title("👁 Eye Strain & Posture Monitor")

col1, col2, col3 = st.columns(3)
blink_box = col1.empty()
ear_box = col2.empty()
dist_box = col3.empty()

status_box = st.empty()

# ==== Video Processor ====
mp_face_mesh = mp.solutions.face_mesh

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.COUNTER = 0
        self.TOTAL = 0
        self.LIFETIME = 0
        self.last_reset = time.time()
        self.bad_distance_start = None
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        ear, iris_dist_cm = None, None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            def idx_pts(idx_list):
                return [(int(lm[i].x * w), int(lm[i].y * h)) for i in idx_list]

            left_eye_pts = idx_pts(LEFT_EYE_IDX)
            right_eye_pts = idx_pts(RIGHT_EYE_IDX)

            leftEAR = eye_aspect_ratio(left_eye_pts)
            rightEAR = eye_aspect_ratio(right_eye_pts)
            ear = (leftEAR + rightEAR) / 2.0

            # Blink detection
            if ear < EYE_AR_THRESH:
                self.COUNTER += 1
            else:
                if self.COUNTER >= EYE_AR_CONSEC_FRAMES:
                    self.TOTAL += 1
                    self.LIFETIME += 1
                self.COUNTER = 0

            iris_dist_cm = iris_distance(lm, LEFT_IRIS_IDX, w, h)

        # Blink reset logic
        elapsed = time.time() - self.last_reset
        if elapsed >= 60:
            if self.TOTAL < MIN_BLINKS_PER_MINUTE:
                status_box.warning(f"⚠️ Low blink rate: {self.TOTAL} blinks in last minute")
            else:
                status_box.success(f"✅ Healthy blink rate: {self.TOTAL} blinks in last minute")
            self.TOTAL = 0
            self.last_reset = time.time()

        # Distance posture logic
        if iris_dist_cm:
            dist_box.metric("Distance (cm)", f"{iris_dist_cm:.1f}")
            if iris_dist_cm < 20 or iris_dist_cm > 80:
                if self.bad_distance_start is None:
                    self.bad_distance_start = time.time()
                elif time.time() - self.bad_distance_start > 60:
                    status_box.warning(f"⚠️ Poor posture for 1 min! ({iris_dist_cm:.1f} cm)")
            else:
                self.bad_distance_start = None
        else:
            dist_box.metric("Distance (cm)", "–")

        # Always update metrics
        blink_box.metric("Blinks (last min)", self.TOTAL, help=f"Lifetime: {self.LIFETIME}")
        ear_box.metric("EAR", f"{ear:.2f}" if ear else "–")

        return img  # Show video with face landmarks (optional)

# ==== Run Streamlit-WebRTC ====
webrtc_streamer(
    key="eye-monitor",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
