import tkinter
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import datetime
from gtts import gTTS
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import os
import threading

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def popupmsg(msg):
    popup=tkinter.Tk()
    popup.wm_title("Urgent")
    style=ttk.Style(popup)
    style.theme_use('classic')
    style.configure('Test.Tlabel',background='aqua')
    label=ttk.Label(popup,text=msg,style='Test.TLabel')
    label.pack(side='top',fill="x",pady=10)
    B1=ttk.Button(popup,text="Okay",command=popup.destroy)
    B1.pack()
    popup.mainloop()

def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=True,
		help="path to facial landmark predictor")
	ap.add_argument("-v", "--video", type=str, default="",
		help="path to input video file")
	args = vars(ap.parse_args())

	# define two constants, one for the eye aspect ratio to indicate
	# blink and then a second constant for the number of consecutive
	# frames the eye must be below the threshold
	EYE_AR_THRESH = 0.3
	EYE_AR_CONSEC_FRAMES = 3
	MIN_BLINKS_PER_MINUTE = 15

	# initialize the frame counters and the total number of blinks
	COUNTER = 0
	TOTAL = 0

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])
	print(type(predictor),predictor)

	# grab the indexes of the facial landmarks for the left and
	# right eye, respectively
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	# --- MODIFICATION START ---
	# Start video stream using OpenCV's VideoCapture for more robust handling
	print("[INFO] accessing video stream...")
	# Use 0 for webcam, or the provided video file path
	video_source = 0 if args["video"] == "" else args["video"]
	vs = cv2.VideoCapture(video_source)

	# Check if the video source was opened correctly
	if not vs.isOpened():
		print(f"Error: Could not open video source: '{video_source}'")
		exit()
	
	time.sleep(1.0)
	# --- MODIFICATION END ---


	# initialize timer for blink rate check
	last_reset_time = time.time()

	# loop over frames from the video stream
	while True:
		# check to see if 60 seconds have passed
		if time.time() - last_reset_time > 60.0:
			if TOTAL < MIN_BLINKS_PER_MINUTE:
				msg = f"Low blink rate detected ({TOTAL}/min). Please rest your eyes!"
				# run popup in a thread to not block the main loop
				popup_thread = threading.Thread(target=popupmsg, args=(msg,))
				popup_thread.daemon = True
				popup_thread.start()
			
			TOTAL = 0
			last_reset_time = time.time()

		# --- MODIFICATION START ---
		# Read a frame from the video capture object
		(ret, frame) = vs.read()

		# 'ret' will be False if a frame could not be read (e.g., end of video)
		if not ret:
			print("Could not read frame. Exiting loop.")
			break
		# --- MODIFICATION END ---
		
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# detect faces in the grayscale frame
		rects = detector(gray, 0)
		
		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			
			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			
			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0
			
			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			
			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1
			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1
				# reset the eye frame counter
				COUNTER = 0
			
			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
		# show the frame
		cv2.imshow("Strain Analyser", frame)
		key = cv2.waitKey(1) & 0xFF
	
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
			
	# do a bit of cleanup
	cv2.destroyAllWindows()
	# --- MODIFICATION START ---
	# Use release() for VideoCapture objects
	vs.release()
	# --- MODIFICATION END ---


if __name__=="__main__":
	main()