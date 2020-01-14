import cv2
import time

def write_text(frame2):
    file = open("testfile.txt", "r")
    val = file.readline()

    if int(val) < 160 and int(val) > 1:
        cv2.putText(frame2, 'injured', (160, 430), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        print('injured')
    elif int(val) > 180:
        cv2.putText(frame2, 'not injured', (160, 430), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        print('not injured')
    elif int(val) == 0:
        cv2.putText(frame2, 'not detected', (160, 430), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)
        print('not detected')
    return frame2

camera = cv2.VideoCapture(0)

while camera.isOpened():
    success, frame = camera.read()

    cv2.imwrite('frame.jpg', frame)

    frame2 = frame
    frame2 = write_text(frame2)

    cv2.imshow('frame', frame2)

    if not success:
        break

    key_code = cv2.waitKey(1)
    if key_code in [27, ord('q')]:
        break
camera.release()
cv2.destroyAllWindows()