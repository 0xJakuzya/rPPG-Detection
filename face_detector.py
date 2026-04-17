import cv2
import mediapipe as mp
import numpy as np
import config
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

class FaceDetector:
    def __init__(self):
        base_options = mp_python.BaseOptions(
            model_asset_path=config.FACE_MODEL_PATH
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=config.FACE_MAX_NUM,
            min_face_detection_confidence=config.FACE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MIN_TRACKING_CONFIDENCE,
            output_face_blendshapes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def get_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        h, w = frame.shape[:2]
        landmarks = [
            (int(p.x * w), int(p.y * h))
            for p in result.face_landmarks[0]
        ]
        return landmarks

    def get_roi(self, frame, landmarks):
        if landmarks is None:
            return None, None, None

        def make_mask(idxs):
            pts = np.array([landmarks[i] for i in idxs], np.int32)
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            return cv2.bitwise_and(frame, frame, mask=mask)

        return make_mask(config.LEFT_CHEEK_IDX), make_mask(config.RIGHT_CHEEK_IDX), make_mask(config.FOREHEAD_IDX)

    def draw_landmarks(self, frame, landmarks):
        if landmarks is None:
            return frame
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame

    def close(self):
        self.landmarker.close()
