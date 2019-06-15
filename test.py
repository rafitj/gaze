import cv2
from gaze_tracking import GazeTracking
import statistics
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import numpy as np
import sys

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
hor_pts = []
vert_pts = []
heatmap_mode = False
frame_count = 0
x_pts = []
y_pts = []


def emotion_detector():

    # parameters for loading data and images
    detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
    emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'
    img_path = sys.argv[1]

    # hyper-parameters for bounding boxes shape
    # loading models
    face_detection = cv2.CascadeClassifier(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

    #reading the frame
    orig_frame = cv2.imread(img_path)
    frame = cv2.imread(img_path,0)
    faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) &amp;amp;amp;gt; 0:
        faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)

    cv2.imshow('test_face', orig_frame)
    cv2.imwrite('test_output/'+img_path.split('/')[-1],orig_frame)

if (cv2.waitKey(2000) &amp;amp;amp;amp; 0xFF == ord('q')):
    sys.exit("Thanks")
cv2.destroyAllWindows()

if len(faces) &amp;amp;amp;gt; 0:
    faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)

cv2.imshow('test_face', orig_frame)
cv2.imwrite('test_output/'+img_path.split('/')[-1],orig_frame)
if (cv2.waitKey(2000) &amp;amp;amp;amp; 0xFF == ord('q')):
    sys.exit("Thanks")
cv2.destroyAllWindows()

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

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    roi_color = cv2.flip(roi_color,1)
    return roi_color


while True:
    _, frame = webcam.read()
    
    gaze.refresh(frame)
    frame = face_detector(frame)
    
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
#    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
#    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Test", frame)

    if cv2.waitKey(1) == 27:
        break
