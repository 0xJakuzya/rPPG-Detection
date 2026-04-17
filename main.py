import cv2

import config
from video_capture import VideoCapture
from face_detector import FaceDetector

cap = VideoCapture()
cap.start()
detector = FaceDetector()

while True:
    frame = cap.read()
    if frame is None:
        continue

    landmarks = detector.get_landmarks(frame)
    frame = detector.draw_landmarks(frame, landmarks)

    cv2.imshow(config.WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
detector.close()
cv2.destroyAllWindows()
