# python face_detect.py \
# --shape-predictor shape_predictor_68_face_landmarks.dat


from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np

# path to facial landmark predictor
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True)
args = vars(ap.parse_args())


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks
(lebStart, lebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rebStart, rebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(jawStart, jawEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
# vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the face coordinates, then use the
        faceline = shape[jawStart:lebEnd]

        # compute the convex hull for face, then
        # visualize each of the face
        facelineHull = cv2.convexHull(faceline)

        mask = np.zeros(frame.shape, dtype='uint8')
        cv2.drawContours(frame, [facelineHull], -1, (0, 0, 0), thickness=cv2.FILLED)
        cv2.drawContours(frame, [facelineHull], -1, (0, 255, 0))
    # show the frame
    cv2.imshow("Frame", frame)
    # cv2.imshow("Frame", mask)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()