import mediapipe as mp
import cv2
mp_iris = mp.solutions.iris
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
with mp_iris.Iris() as iris:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = iris.process(image)

    # Draw the eye and iris annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.face_landmarks_with_iris:
        mp_draw.draw_iris_landmarks(image,results.face_landmarks_with_iris)
    cv2.imshow('MediaPipe Iris', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()