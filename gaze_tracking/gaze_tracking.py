from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration


class GazeTracking(object):
    """
    This class tracks the user's gaze and records location of gaze.
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # pre-trained _face_detector model is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # pre-trained _predictor model is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Verify that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it."""
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_left.pupil.x
            y = self.eye_right.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_left_coords_plot(self):
        """Returns the coordinates of the left pupil plotting"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.int_x
            y = self.eye_left.origin[1] + self.eye_left.pupil.int_y

            return (x, y)

    def pupil_right_coords_plot(self):
        """Returns the coordinates of the right pupil plotting"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_left.pupil.int_x
            y = self.eye_right.origin[1] + self.eye_left.pupil.int_y
            
            return (x, y)

    def normalize_horizontal(self, x):
        return (x-0.53)/(0.68-0.53)
    
    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_l_x = self.eye_left.pupil.x
            pupil_r_x = self.eye_right.pupil.x
            pupil_left = pupil_l_x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = pupil_r_x / (self.eye_right.center[0] * 2 - 10)
            horizontal_gaze = (pupil_left + pupil_right) / 2
            return self.normalize_horizontal(horizontal_gaze)

    def normalize_vertical(self, y):
        return (y-0.69)/(0.795-0.69)
    
    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_l_y = self.eye_left.pupil.y
            pupil_r_y = self.eye_right.pupil.y
            pupil_left = pupil_l_y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = pupil_r_y  / (self.eye_right.center[1] * 2 - 10)
            vertical_gaze = (pupil_left + pupil_right) / 2
            return self.normalize_vertical(vertical_gaze)

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords_plot()
            x_right, y_right = self.pupil_right_coords_plot()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
