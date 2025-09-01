import argparse
import time
import dlib
import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream, FileVideoStream

# compute EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
    ap.add_argument("-v", "--video", type=str, default="",
        help="path to input video file (default: webcam)")
    args = vars(ap.parse_args())

    # EAR thresholds
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 3
    COUNTER, TOTAL = 0, 0

    print("[INFO] loading facial landmark predictor...")
    predictor = dlib.shape_predictor(args["shape_predictor"])

    # landmark indexes for eyes
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # use OpenCV Haar cascade instead of dlib detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # video source
    if args["video"] == "":
        print("[INFO] starting webcam...")
        vs = VideoStream(src=0).start()
        fileStream = False
    else:
        print("[INFO] opening video file...")
        vs = FileVideoStream(args["video"]).start()
        fileStream = True
    time.sleep(1.0)

    while True:
        if fileStream and not vs.more():
            break

        frame = vs.read()
        if frame is None or not isinstance(frame, np.ndarray):
            continue

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces with Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            # convert cv2 rectangle to dlib rectangle
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            # landmark prediction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            shape = predictor(rgb, rect)

            #shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

            cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Strain Analyser", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
