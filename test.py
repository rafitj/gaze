import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
hor_pts = []
vert_pts = []


while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    overlay = frame.copy()
    text = ""
    

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    if (gaze.horizontal_ratio()!= None and gaze.vertical_ratio() != None):
        hor_pts.append(int(1280*(1-gaze.horizontal_ratio())))
        vert_pts.append(int(720*gaze.vertical_ratio()))

    for x, y in zip(hor_pts, vert_pts):
        cv2.circle(overlay, (x,y), 10, (0,0,255), 20)
        alpha = 0.01
        frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)


    corner = [(1280,720),(0,720),(1280,0),(0,0)]

    for coord in corner:
        cv2.circle(frame, coord, 50, (0,0,255), 5)

    horizontal_ratio = str(gaze.horizontal_ratio())
    vertical_ratio = str(gaze.vertical_ratio())

    cv2.putText(frame, horizontal_ratio, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (147, 58, 31), 2)
    cv2.putText(frame, vertical_ratio, (90, 105), cv2.FONT_HERSHEY_DUPLEX, 1.5, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Test", frame)

    if cv2.waitKey(1) == 27:
        break
