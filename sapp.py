import streamlit as st
import cv2
import time
import numpy as np
import mediapipe as mp
import math
import pandas as pd
from plyer import notification

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX = [468, 469, 470, 471]
IRIS_DIAMETER_MM = 11.7
FOCAL_LENGTH_PX = 600  # calibrated

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

def process_face_landmarks(landmarks, w, h):
    """Extracts EAR and iris distance from face landmarks."""
    def idx_pts(idx_list):
        return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idx_list]

    left_eye_pts = idx_pts(LEFT_EYE_IDX)
    right_eye_pts = idx_pts(RIGHT_EYE_IDX)

    leftEAR = eye_aspect_ratio(left_eye_pts)
    rightEAR = eye_aspect_ratio(right_eye_pts)
    ear = (leftEAR + rightEAR) / 2.0

    iris_dist_cm = iris_distance(landmarks, LEFT_IRIS_IDX, w, h)
    
    return ear, iris_dist_cm

def update_blink_counter(ear, counter, total):
    """Updates blink counter based on EAR."""
    if ear < EYE_AR_THRESH:
        counter += 1
    elif counter >= EYE_AR_CONSEC_FRAMES:
        total += 1
        counter = 0
    else:
        counter = 0
    return counter, total

# Streamlit UI
st.title("👁 Eye Strain & Distance Monitor")
st.write("Live blink counter + eye-to-camera distance")

FRAME_WINDOW = st.image([])
blink_counter = st.empty()
distance_text = st.empty()
avg_blink_rate_display = st.empty()
distance_box = st.empty()
blink_box = st.empty()
status_box = st.empty()   # for success/warning messages
cap = cv2.VideoCapture(0)

COUNTER = 0
TOTAL = 0
last_reset = time.time()
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3
MIN_BLINKS_PER_MINUTE = 15
RE_ALERT_INTERVAL_SEC = 30 # Re-alert every 30 seconds
# Extra state
bad_distance_start = None
last_alert_time = None
BLINK_HISTORY = []

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear = None
        iris_dist_cm = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear, iris_dist_cm = process_face_landmarks(landmarks, w, h)
            COUNTER, TOTAL = update_blink_counter(ear, COUNTER, TOTAL)

        # Reset blink count every 60s
        elapsed = time.time() - last_reset
        if elapsed >= 60:
            BLINK_HISTORY.append(TOTAL)
            if TOTAL < MIN_BLINKS_PER_MINUTE:
                blink_box.warning(f"⚠️ Low blink rate: {TOTAL} blinks in last minute")
            else:
                blink_box.success(f"✅ Healthy blink rate: {TOTAL} blinks in last minute")
            TOTAL = 0
            last_reset = time.time()

        # Distance checks
        if iris_dist_cm:
            distance_text.info(f"Distance: {iris_dist_cm:.1f} cm")   # always show distance
            is_bad_posture = iris_dist_cm < 40 or iris_dist_cm > 80

            if is_bad_posture:
                if bad_distance_start is None:
                    bad_distance_start = time.time()
                
                if time.time() - bad_distance_start > 10:
                    status_box.warning(f"⚠️ Poor posture for over 10 seconds! ({iris_dist_cm:.1f} cm)")
                    
                    # Check if it's time for a new alert (initial or re-alert)
                    if last_alert_time is None or (time.time() - last_alert_time > RE_ALERT_INTERVAL_SEC):
                        #st.toast(f"🚨 Poor posture detected! Please adjust your distance.", icon="📏")
                        # Use plyer for a system-level notification
                        notification.notify(
                            title="Eye Strain Monitor",
                            message=f"Poor posture detected! Please adjust your distance.",
                            app_name="Eye Strain Monitor"
                        )
                        last_alert_time = time.time()
            else:
                # Posture is good, reset timers and clear warnings
                bad_distance_start = None
                last_alert_time = None
                status_box.empty()  # Clear the warning when posture is good


        blink_counter.info(f"Total Blinks: {TOTAL}")

        # Update the average blink rate display on every frame
        if BLINK_HISTORY:
            last_n_mins_history = BLINK_HISTORY[-5:]
            num_mins = len(last_n_mins_history)
            avg_blinks = sum(last_n_mins_history) / num_mins
            avg_blink_rate_display.info(f"Avg blink rate (last {num_mins} mins): {avg_blinks:.1f} blinks/min")

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
