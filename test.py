import cv2
from gaze_tracking import GazeTracking
import statistics

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
hor_pts = []
vert_pts = []
heatmap_mode = False
frame_count = 0
x_pts = []
y_pts = []

def indicator(gaze):
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    return text

while True:
    _, frame = webcam.read()
    
    gaze.refresh(frame)
    
    frame = gaze.annotated_frame()
    overlay = frame.copy()
    
    frame_count += 1
    frame_refresh = True if (frame_count % 10 is 0) else False
    
    if (gaze.horizontal_ratio()!= None and gaze.vertical_ratio() != None):
        temp_x = int(1280*(1-gaze.horizontal_ratio()))
        temp_y = int(720*gaze.vertical_ratio())
        x_pts.append(temp_x)
        y_pts.append(temp_y)
        if (len(x_pts)>10 or len(y_pts)>10):
            del x_pts[0]
            del y_pts[0]
    
    avg_x = int(statistics.mean(x_pts)) if x_pts else None
    avg_y = int(statistics.mean(y_pts)) if y_pts else None

    if frame_refresh:
        if (avg_x != None and avg_y != None):
            hor_pts.append(avg_x)
            vert_pts.append(avg_y)

    if (heatmap_mode):
        for x, y in zip(hor_pts, vert_pts):
            cv2.circle(overlay, (x,y), 10, (0,0,255), 20)
            alpha = 0.01
            frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    
    else:
        if (avg_x != None and avg_y != None):
            cv2.circle(overlay, (avg_x,avg_y), 40, (0,0,255), 80)
            alpha = 0.6
            frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    


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
