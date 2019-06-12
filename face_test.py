import cv2
import numpy as np

webcam = cv2.VideoCapture(0)


while True:
    _, frame = webcam.read()
    
    overlay = frame.copy()
    kernel = np.ones((3, 3), np.uint8)
    new_frame = cv2.bilateralFilter(overlay, 10, 15, 15)
    new_frame = cv2.erode(new_frame, kernel, iterations=3)
#    new_frame = cv2.threshold(new_frame, 50, 255, cv2.THRESH_BINARY)[1]


    cv2.imshow("Test", new_frame)

    if cv2.waitKey(1) == 27:
        break
