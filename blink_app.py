import sys
import time
import math
import cv2
import numpy as np
import mediapipe as mp
from PyQt5 import QtWidgets, QtGui, QtCore
from plyer import notification

# === Mediapipe setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# === Config ===
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_IDX = [468, 469, 470, 471]

BLINK_THRESH = 0.27
CONSEC_FRAMES = 3
BLINK_MIN_HEALTHY = 15
BLINK_MIN_HIGH_RISK = 5

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(eye_pts):
    A = euclidean(eye_pts[1], eye_pts[5])
    B = euclidean(eye_pts[2], eye_pts[4])
    C = euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C) if C != 0 else 0

def iris_distance(landmarks, iris_idx, w, h, K=None):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in iris_idx]
    xs = [p[0] for p in pts]
    iris_px = max(xs) - min(xs)
    if iris_px <= 0:
        return None
    if K:  # calibrated
        return K / iris_px
    return None

# === PyQt Application ===
class BlinkApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("👁 Eye Strain Monitor")
        self.setGeometry(200, 200, 400, 250)

        layout = QtWidgets.QVBoxLayout()

        self.label_blinks = QtWidgets.QLabel("Lifetime blinks: 0")
        self.label_rate = QtWidgets.QLabel("Blinks this minute: 0")
        self.label_ear = QtWidgets.QLabel("EAR: –")
        self.label_dist = QtWidgets.QLabel("Distance: –")
        self.label_status = QtWidgets.QLabel("Status: OK")

        # Distance input + calibration
        self.dist_input = QtWidgets.QSpinBox()
        self.dist_input.setRange(20, 200)   # Allow 20cm–200cm
        self.dist_input.setValue(50)        # Default = 50 cm
        layout.addWidget(QtWidgets.QLabel("Enter your current distance (cm):"))
        layout.addWidget(self.dist_input)

        self.button_calibrate = QtWidgets.QPushButton("Calibrate")
        self.button_calibrate.clicked.connect(self.calibrate)
        layout.addWidget(self.button_calibrate)



        for lbl in [self.label_blinks, self.label_rate, self.label_ear, self.label_dist, self.label_status]:
            lbl.setFont(QtGui.QFont("Arial", 12))
            layout.addWidget(lbl)

        self.setLayout(layout)

        # Blink state
        self.counter = 0
        self.lifetime = 0
        self.minute_count = 0
        self.last_minute = time.time()

        self.bad_distance_start = None
        self.K = None  # calibration constant

        # Timer for video loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def calibrate(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            xs = [int(lm[i].x * w) for i in LEFT_IRIS_IDX]
            iris_px = max(xs) - min(xs)
            if iris_px > 0:
                d_known = self.dist_input.value()  # user input in cm
                self.K = d_known * iris_px
                self.notify(f"✅ Calibration done at {d_known} cm")


    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear, dist_cm = None, None

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            def pts(idx_list):
                return [(int(lm[i].x * w), int(lm[i].y * h)) for i in idx_list]

            left_eye = pts(LEFT_EYE_IDX)
            right_eye = pts(RIGHT_EYE_IDX)

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            if ear < BLINK_THRESH:
                self.counter += 1
            else:
                if self.counter >= CONSEC_FRAMES:
                    self.lifetime += 1
                    self.minute_count += 1
                self.counter = 0

            dist_cm = iris_distance(lm, LEFT_IRIS_IDX, w, h, self.K)

        # Per-minute reset
        elapsed = time.time() - self.last_minute
        if elapsed >= 60:
            if self.minute_count < BLINK_MIN_HEALTHY:
                if self.minute_count < BLINK_MIN_HIGH_RISK:
                    self.notify(f"🔴 High strain: only {self.minute_count} blinks/min")
                else:
                    self.notify(f"⚠️ Low blink rate: {self.minute_count} blinks/min")
            self.minute_count = 0
            self.last_minute = time.time()

        # Distance check
        if dist_cm:
            if dist_cm < 30 or dist_cm > 100:
                if self.bad_distance_start is None:
                    self.bad_distance_start = time.time()
                elif time.time() - self.bad_distance_start > 60:
                    self.notify(f"⚠️ Poor posture >60s ({dist_cm:.1f} cm)")
                    self.bad_distance_start = time.time() + 5
            else:
                self.bad_distance_start = None

        # Update labels
        self.label_blinks.setText(f"Lifetime blinks: {self.lifetime}")
        self.label_rate.setText(f"Blinks this minute: {self.minute_count}")
        self.label_ear.setText(f"EAR: {ear:.2f}" if ear else "EAR: –")
        self.label_dist.setText(f"Distance: {dist_cm:.1f} cm" if dist_cm else "Distance: –")
        self.label_status.setText("Status: Monitoring...")

    def notify(self, msg):
        notification.notify(
            title="Eye Strain Monitor",
            message=msg,
            timeout=5
        )

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = BlinkApp()
    win.show()
    sys.exit(app.exec_())
