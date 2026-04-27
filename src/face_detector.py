import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from src import config

class FaceDetector:
    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path=config.FACE_MODEL_PATH)
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
        return [
            (int(p.x * w), int(p.y * h))
            for p in result.face_landmarks[0]
        ]

    def get_roi(self, frame, landmarks):
        def make_mask(idxs, crop_top_frac=None, crop_bottom_frac=None):
            pts = np.array([landmarks[i] for i in idxs], np.int32)
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            if crop_top_frac is not None or crop_bottom_frac is not None:
                y_min = int(pts[:, 1].min())
                y_max = int(pts[:, 1].max())
                h_roi = y_max - y_min
                if crop_top_frac is not None:
                    mask[:y_min + int(h_roi * crop_top_frac)] = 0
                if crop_bottom_frac is not None:
                    mask[y_min + int(h_roi * crop_bottom_frac):] = 0
            return cv2.bitwise_and(frame, frame, mask=mask)
        forehead = make_mask(config.FOREHEAD_IDX, crop_bottom_frac=0.40)
        left_cheek = make_mask(config.LEFT_CHEEK_IDX)
        right_cheek = make_mask(config.RIGHT_CHEEK_IDX)
        return left_cheek, right_cheek, forehead

    def draw_landmarks(self, frame, landmarks):
        if landmarks is None:
            return frame
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame

    def close(self):
        self.landmarker.close()
