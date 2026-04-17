CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
FPS_TARGET: int = 30

WINDOW_NAME: str = "rPPG"
FONT_SCALE: float = 0.6
FONT_COLOR: tuple = (0, 255, 0)
FONT_THICKNESS: int = 1

FACE_MODEL_PATH: str = 'face_landmarker.task'
FACE_MAX_NUM: int = 1
FACE_MIN_DETECTION_CONFIDENCE: float = 0.5
FACE_MIN_TRACKING_CONFIDENCE: float = 0.5

LEFT_CHEEK_IDX: list = [50, 101, 118, 117, 116, 123, 147, 213, 192, 214]
RIGHT_CHEEK_IDX: list = [280, 330, 347, 346, 345, 352, 376, 433, 416, 434]
FOREHEAD_IDX: list = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                      361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                      176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                      162, 21, 54, 103, 67, 109, 10]
