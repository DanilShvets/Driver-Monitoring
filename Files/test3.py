import cv2
import numpy as np
import dlib
from math import hypot
# import pyglet
import time

cap = cv2.VideoCapture(0)
board = np.zeros((300, 1400), np.uint8)
board[:] = 255

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

keyboard = np.zeros((600, 1000, 3), np.uint8)
key_set_1 = {}
key_set_2 = {}
keys1 = list('QWEERTASDFGZXCV<')
keys2 = list('YUIOPHJKL_VBNM<')
for i in range(15):
    key_set_1[i] = keys1[i]
    key_set_2[i] = keys2[i]

def draw_letters(letter_index, text, letter_light):
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200
    elif letter_index == 10:
        x = 0
        y = 400
    elif letter_index == 11:
        x = 200
        y = 400
    elif letter_index == 12:
        x = 400
        y = 400
    elif letter_index == 13:
        x = 600
        y = 400
    elif letter_index == 14:
        x = 800
        y = 400
    width = 200
    height = 200
    th = 3

    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getSize(text, font_letter, font_scale, font_th)[0]
    text_width, text_height = text_size[0], text_size[1]
    text_x = int((width - text_width)/2) + x
    text_y = int((height + text_height) / 2) + y

    if letter_light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)

    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)

font = cv2.FONT_HERSHEY_PLAIN

def draw_menu():
    rows, cols, _ = keyboard.shape
    th_lines = 4
    cv2.line(keyboard, (int(cols/2) - int(th_lines), 0), (int(cols/2) - int(th_lines/2), rows), (51, 51, 51), th_lines)
    cv2.putText(keyboard, "LEFT", (80, 300), font, 6, (255, 255, 255, 5))
    cv2.putText(keyboard, "RIGHT", (80 + int(cols/2), 300), font, 6, (255, 255, 255, 5))

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2, int((p1.y + p2.y)/2))

def get_blink_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36,42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42,48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.uint8)
    right_eye = np.array(right_eye, np.uint8)
    return left_eye, right_eye

def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),\
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),\
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),\
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),\
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),\
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)])


while True:

    _, frame = cap.read()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()