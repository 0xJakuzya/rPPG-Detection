from collections import deque

import cv2
import numpy as np

from models.pos import pos

from src import config
from src.face_detector import FaceDetector
from src.utils import (
    estimate_hr,
    extract_mean_rgb_from_patches,
    extract_multi_rois_patches,
    make_patch_preview,
)
from src.video import VideoCapture
from src.visualization import draw_status


def run_tester() -> None:
    camera = VideoCapture()
    camera.start()

    detector = FaceDetector()
    fps = float(config.FPS_TARGET)
    min_frames = max(int(fps * 2), 2)
    rgb_buffer = deque(maxlen=config.PHYSNET_WINDOW)

    while True:
        frame = camera.read()
        if frame is None:
            continue

        display = frame.copy()
        landmarks = detector.get_landmarks(frame)
        heart_rate = None
        patch_preview = None

        if landmarks is None:
            rgb_buffer.clear()
            status, color = "NO FACE", (0, 0, 255)
        else:
            detector.draw_landmarks(display, landmarks)
            patches = extract_multi_rois_patches(detector, frame, landmarks)
            patch_preview = make_patch_preview(patches)
            patch_rgb = extract_mean_rgb_from_patches(patches)
            if patch_rgb is not None:
                rgb_buffer.append(patch_rgb.tolist())
            if len(rgb_buffer) >= min_frames:
                rgb_trace = np.asarray(rgb_buffer, dtype=np.float32)
                predicted_bvp = pos(rgb_trace, fps)
                result = estimate_hr(predicted_bvp, fps)
                heart_rate = round(float(result)) if result is not None else None
            status, color = "DETECTED", (0, 255, 0)

        draw_status(display, heart_rate, status, color)
        cv2.imshow(config.WINDOW_NAME, display)
        if patch_preview is not None:
            cv2.imshow("ROI patches", patch_preview)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.stop()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_tester()
