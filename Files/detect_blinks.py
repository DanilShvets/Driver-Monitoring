# source work/bin/activate
# cd work
# python detect_blinks.py \
# --shape-predictor shape_predictor_68_face_landmarks.dat

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
from datetime import datetime
from playsound import playsound
import matplotlib
import matplotlib.pyplot as plt

w = 1280     # ширина окна
h = 720
counter_arr_eye = [0, 0]
ear_arr = []

def eye_aspect_ratio(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def mouth_aspect_ratio(mou):

    A1 = dist.euclidean(mou[1], mou[7])
    B1 = dist.euclidean(mou[2], mou[6])
    C1 = dist.euclidean(mou[3], mou[5])
    D1 = dist.euclidean(mou[0], mou[4])

    mou = (A1 + B1 + C1) / (2.0 * D1)

    return mou


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 2


COUNTER = 0
TOTAL = 0

video = cv2.VideoCapture(0)

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
first_detected = datetime.now()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)
fps = video.get(cv2.CAP_PROP_FPS)
# print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


while True:

    if fileStream and not vs.more():
        break

    frame = vs.read()
    frame = imutils.resize(frame, width=w, height=h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouthEAR = mouth_aspect_ratio(mouth)

        ear = (leftEAR + rightEAR) / 2.0
        mou = mouthEAR

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)


        ear_arr.append(ear)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER > 25:
                playsound('sound.mp3')
                cv2.putText(frame, "WAKE UP", (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            counter_arr_eye.append(COUNTER)
            COUNTER = 0

        cv2.putText(frame, "Count: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mou), (w - 150, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Camera", frame)
    # key = cv2.waitKey(1) & 0xFF
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
last_detected = datetime.now()
t = (last_detected - first_detected).total_seconds()

plt.rcParams['font.size'] = '16'
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
x = np.linspace(0, t, len(ear_arr))
plt.plot(x, ear_arr)
plt.title('График зависимости EAR от времени')
plt.xlabel("Время, с", fontsize=16)
plt.ylabel('EAR', fontsize=16)
plt.grid(True)
plt.show()





