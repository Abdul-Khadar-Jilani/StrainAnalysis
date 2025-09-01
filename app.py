# filename: app.py
import streamlit as st
import time
import math
import numpy as np
import cv2
import mediapipe as mp
import threading # Import the threading library
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

# --- MediaPipe and constants ---
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX = [468, 469, 470, 471]
IRIS_DIAMETER_MM = 11.7
FOCAL_LENGTH_PX = 600

# --- Helper Functions (no changes) ---
def euclidean(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def eye_aspect_ratio(eye_pts):
    A = euclidean(eye_pts[1], eye_pts[5])
    B = euclidean(eye_pts[2], eye_pts[4])
    C = euclidean(eye_pts[0], eye_pts[3])
    if C == 0: return 0.0
    return (A + B) / (2.0 * C)

def iris_distance(landmarks, iris_idx, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in iris_idx]
    xs = [p[0] for p in pts]
    iris_d_px = max(xs) - min(xs)
    if iris_d_px == 0: return None
    return ((FOCAL_LENGTH_PX * IRIS_DIAMETER_MM) / iris_d_px) / 10.0

def draw_eye_contour(frame, pts):
    pts_arr = np.array(pts, dtype=np.int32)
    cv2.polylines(frame, [pts_arr], isClosed=True, thickness=1, color=(0, 255, 0))

# --- Session State Initialization with Lock ---
if 'lock' not in st.session_state:
    st.session_state.lock = threading.Lock()
    st.session_state.metrics = {
        "blinks_in_minute": 0,
        "ear": 0.0,
        "distance": 0.0,
        "last_reset": time.time(),
        "blink_counter": 0, # Frames below EAR threshold
        "too_close_counter": 0, # Frames where user is too close
    }

# --- The Core Video Processing Class ---
class StrainAnalyserTransformer(VideoTransformerBase):
    def __init__(self, ear_thresh, consec_frames, min_blinks_per_min, min_dist_cm, close_consec_frames):
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.ear_thresh = ear_thresh
        self.consec_frames = consec_frames
        self.min_blinks_per_min = min_blinks_per_min
        self.min_dist_cm = min_dist_cm
        self.close_consec_frames = close_consec_frames

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        # Use the lock to safely access and modify session state
        with st.session_state.lock:
            metrics = st.session_state.metrics
            
            # Check for minute reset for blink count
            if time.time() - metrics["last_reset"] >= 60.0:
                if metrics["blinks_in_minute"] < self.min_blinks_per_min:
                    msg = f"Low blink rate: {metrics['blinks_in_minute']} blinks in the last minute!"
                    st.toast(f"🚨 {msg}", icon="👀")
                metrics["blinks_in_minute"] = 0
                metrics["last_reset"] = time.time()

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                def idx_pts(idx_list):
                    return [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in idx_list]

                left_eye_pts = idx_pts(LEFT_EYE_IDX)
                right_eye_pts = idx_pts(RIGHT_EYE_IDX)
                
                draw_eye_contour(img, left_eye_pts)
                draw_eye_contour(img, right_eye_pts)

                # --- Blink Detection Logic ---
                ear = (eye_aspect_ratio(left_eye_pts) + eye_aspect_ratio(right_eye_pts)) / 2.0
                metrics["ear"] = ear
                if ear < self.ear_thresh:
                    metrics["blink_counter"] += 1
                else:
                    if metrics["blink_counter"] >= self.consec_frames:
                        metrics["blinks_in_minute"] += 1
                    metrics["blink_counter"] = 0

                # --- Distance Detection Logic ---
                dist = iris_distance(face_landmarks, LEFT_IRIS_IDX, w, h)
                if dist is not None:
                    metrics["distance"] = dist
                    if dist < self.min_dist_cm:
                        metrics["too_close_counter"] += 1
                    else:
                        metrics["too_close_counter"] = 0
                    
                    # If user is too close for a set number of frames, show a warning
                    if metrics["too_close_counter"] == self.close_consec_frames:
                        st.toast(f"🚨 Too close! Move back from the screen.", icon="📏")

                    cv2.putText(img, f"Dist: {dist:.1f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Draw HUD
            cv2.putText(img, f"Blinks: {metrics['blinks_in_minute']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, f"EAR: {metrics['ear']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
st.set_page_config(page_title="Eye Strain Monitor", page_icon="👀", layout="wide")
st.title("Real-Time Eye Strain Monitor 👀")
st.write("This app uses your webcam to monitor your blink rate and distance from the screen to help prevent eye strain.")

with st.sidebar:
    st.header("⚙️ Configuration")
    EAR_THRESH = st.slider("EAR Threshold", 0.0, 0.4, 0.27, 0.01)
    CONSEC_FRAMES = st.slider("Blink Consecutive Frames", 1, 10, 3, 1)
    MIN_BLINKS = st.slider("Min. Blinks per Minute", 5, 30, 15, 1)
    st.markdown("---")
    MIN_DIST = st.slider("Min. Distance (cm)", 10, 60, 30, 1)
    CLOSE_CONSEC_FRAMES = st.slider("'Too Close' Consecutive Frames", 5, 30, 10, 1)

# Main app layout
col1, col2 = st.columns([3, 1])

with col1:
    st.header("Webcam Feed")
    ctx = webrtc_streamer(
        key="strain-analyser",
        video_processor_factory=lambda: StrainAnalyserTransformer(
            ear_thresh=EAR_THRESH,
            consec_frames=CONSEC_FRAMES,
            min_blinks_per_min=MIN_BLINKS,
            min_dist_cm=MIN_DIST,
            close_consec_frames=CLOSE_CONSEC_FRAMES
        ),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Auto-refresh to update metrics
st_autorefresh(interval=200, limit=None, key="freshener")

with col2:
    st.header("📊 Live Metrics")
    blinks_metric = st.empty()
    ear_metric = st.empty()
    dist_metric = st.empty()
    
    # Read metrics from session state using the lock
    with st.session_state.lock:
        metrics = st.session_state.metrics
        blinks_metric.metric("Blinks (This Minute)", f"{metrics['blinks_in_minute']}")
        ear_metric.metric("Eye Aspect Ratio (EAR)", f"{metrics['ear']:.2f}")
        dist_metric.metric("Screen Distance (cm)", f"{metrics['distance']:.1f}")