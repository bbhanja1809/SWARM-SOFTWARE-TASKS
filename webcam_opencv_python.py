import cv2 as cv
from matplotlib import pyplot as plt


# functio to access video of USB connected device
def Mobile_video(num):
    # num value needs tinkering according to the device number
    # video capturing via opencv
    cap = cv.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        cv.imshow('Mobile_Cam',frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

Mobile_video(0)