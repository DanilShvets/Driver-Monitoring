# source work/bin/activate
# cd work
# python readVideo2.py --video {path to the file} \
# --shape-predictor shape_predictor_68_face_landmarks.dat

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import FPS
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

# w = 660     # ширина окна
# h = 720
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
# ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
ap.add_argument("-v", "--video", required=True, help="path to input video file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 2

LC_counter = 0
COUNTER = 0
TOTAL = 0

video = cv2.VideoCapture(0)

# w = int(video.get(3) / 2)
# h = int(video.get(4) / 2)

w = 376
h = 480

print(w, h)

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
#
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(imStart, imEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]



print("[INFO] starting video file thread...")
first_detected = datetime.now()
vs = FileVideoStream(args["video"]).start()
time.sleep(1.0)
# start the FPS timer
fps = FPS().start()
fileStream = True


while True:

    if fileStream and not vs.more():
        break

    # empty_window = cv2.imread('empty_window.jpg', -1)
    # empty_window = cv2.resize(empty_window, (int(w/2), int(h/2)))
    frame = vs.read()
    frame = imutils.resize(frame, width=w, height=h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.normalize(frame, frame, 0, 300, cv2.NORM_MINMAX)
    frame = np.dstack([frame, frame, frame])
    rects = detector(gray, 0)

    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[45:46]
        cv2.circle(frame, (leftEye[0][0], leftEye[0][1]), 1, (0, 0, 255), -1)
        rightEye = shape[36:37]
        cv2.circle(frame, (rightEye[0][0], rightEye[0][1]), 1, (0, 0, 255), -1)
        mouthLeft = shape[48:49]
        cv2.circle(frame, (mouthLeft[0][0], mouthLeft[0][1]), 1, (0, 0, 255), -1)
        mouthRight = shape[54:55]
        cv2.circle(frame, (mouthRight[0][0], mouthRight[0][1]), 1, (0, 0, 255), -1)
        # nose = shape[33:34]
        nose = shape[30:31]
        cv2.circle(frame, (nose[0][0], nose[0][1]), 1, (0, 0, 255), -1)
        chin = shape[8:9]
        cv2.circle(frame, (chin[0][0], chin[0][1]), 1, (0, 0, 255), -1)

        image_points = np.array([
            (nose[0][0], nose[0][1]),  # Nose tip
            (chin[0][0], chin[0][1]),  # Chin
            (rightEye[0][0], rightEye[0][1]),  # Left eye left corner
            (leftEye[0][0], leftEye[0][1]),  # Right eye right corne
            (mouthLeft[0][0], mouthLeft[0][1]),  # Left Mouth corner
            (mouthRight[0][0], mouthRight[0][1])  # Right mouth corner
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -160.0, -45.0),  # Chin
            (-95.0, 45.0, -55.0),  # Left eye left corner
            (95.0, 45.0, -55.0),  # Right eye right corner
            (-55.0, -80.0, -45.0),  # Left Mouth corner
            (55.0, -80.0, -45.0)  # Right mouth corner

        ])

        # cv2.circle(frame, (0, 0), 1, (0, 0, 255), -1)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector,
                                                         camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        innerMouth = shape[imStart:imEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouthEAR = mouth_aspect_ratio(innerMouth)

        ear = (leftEAR + rightEAR) / 2.0
        mou = mouthEAR

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        innerMouthHull = cv2.convexHull(innerMouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [innerMouthHull], -1, (0, 255, 0), 1)

        ear_arr.append(ear)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER > 25:
                playsound('sound.mp3')
                cv2.putText(frame, "WAKE UP", (int(w/2), int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            counter_arr_eye.append(COUNTER)
            COUNTER = 0

        if p2[0] < 200 or p2[1] > 300:
            print(LC_counter)
            LC_counter += 1
            if LC_counter > 90:
                cv2.putText(frame, "DANGER", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        else:
            LC_counter = 0

        cv2.putText(frame, "Count: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mou), (w - 150, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "LC_x: {:.2f}".format(p2[0]), (w - 150, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "LC_y: {:.2f}".format(p2[1]), (w - 150, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # numpy_horizontal = np.hstack((frame, empty_window))

    cv2.imshow("Frame", frame)

    # text_count = cv2.putText(empty_window, "Count: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # cv2.imshow('Frame', numpy_horizontal)

    # key = cv2.waitKey(1) & 0xFF
    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup


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

x_arr_read = []
f_x = open('xArray.txt')
with f_x as infile:
    for line in infile:
        line = line.rstrip('\n')[1:-1]  # this removes first and last parentheses from the line
        x_arr_read.append([float(v) for v in line.split(', ')])
x_arr = x_arr_read[0]
f_x.close()

y_arr_read = []
f_y = open('yArray.txt')
with open('yArray.txt') as infile:
    for line in infile:
        line = line.rstrip('\n')[1:-1]  # this removes first and last parentheses from the line
        y_arr_read.append([float(v) for v in line.split(', ')])
y_arr = y_arr_read[0]
f_y.close()

z_arr_read = []
f_z = open('zArray.txt')
with f_z as infile:
    for line in infile:
        line = line.rstrip('\n')[1:-1]  # this removes first and last parentheses from the line
        z_arr_read.append([float(v) for v in line.split(', ')])
z_arr = z_arr_read[0]
f_z.close()

xAxisArray = np.linspace(0, 0.1*len(x_arr), len(x_arr))

fig, ax = plt.subplots()
ax.plot(xAxisArray, x_arr, label='Отклонение по X')
ax.plot(xAxisArray, y_arr, label='Отклонение по Y')
ax.plot(xAxisArray, z_arr, label='Отклонение по Z')
plt.title('Данные с гироскопа')
plt.xlabel('Время, с')
plt.ylabel('Отклонение')
plt.grid(True)
ax.legend()
plt.show()


