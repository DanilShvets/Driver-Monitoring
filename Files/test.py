import cv2
import datetime

cap = cv2.VideoCapture(0)   # видео с фронталки
# cap = cv2.VideoCapture('video.avi')
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # разрешение камеры
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # разрешение камеры

w = (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        text = str(datetime.datetime.now())
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (10, 50), font, 1, (0, 0, 0), 2)
        cv2.putText(frame, 'Width: ' + str(w) + ', Height: ' + str(h), (10, 150), font, 1, (0, 0, 0), 1)
        # out.write(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # делаем картинку серой
        cv2.imshow('Camera', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
# out.release()
cv2.destroyAllWindows()
