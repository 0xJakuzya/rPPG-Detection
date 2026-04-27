# ── Algorithm
RPPG_METHOD: str = "POS"
FILTER_TYPE: str = "chebyshev2"

# ── Camera 
CAMERA_INDEX: int = 0         
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
FPS_TARGET: int = 30

# Window
WINDOW_NAME: str = "rPPG"
PLOT_H: int = 80               

# Overlay text 
FONT_SCALE: float = 0.55
FONT_COLOR: tuple[int, int, int] = (180, 180, 180)
FONT_THICKNESS: int = 1

#  Signal buffer 
BUFFER_SEC: int = 10      

#  Detrend filter 
DETREND_LAMBDA: float = 100.0 

# ── Chebyshev Type II bandpass
CHEBY_LO: float = 0.7          
CHEBY_HI: float = 3.5          
CHEBY_ORDER: int = 2           
CHEBY_RS: float = 40.0        

# ── HR estimation 
HR_LO_HZ: float = 0.67         
HR_HI_HZ: float = 3.0         

# MediaPipe face model 
FACE_MODEL_PATH: str = "face_landmarker.task"
FACE_MAX_NUM: int = 1
FACE_MIN_DETECTION_CONFIDENCE: float = 0.5
FACE_MIN_TRACKING_CONFIDENCE: float = 0.5

# ── ROI landmark indices 
LEFT_CHEEK_IDX: list[int] = [
    215, 138, 135, 210, 212, 57, 216, 207, 192,
    116, 111, 117, 118, 119, 100, 47, 126, 101,
    123, 137, 177, 50, 36, 209, 129, 205, 147, 187, 206, 203,
]
RIGHT_CHEEK_IDX: list[int] = [
    435, 427, 416, 364, 394, 422, 287, 410, 434, 436,
    349, 348, 347, 346, 345, 447, 323, 280, 352,
    330, 371, 358, 423, 426, 425, 411, 376,
]
FOREHEAD_IDX: list[int] = [
    10, 151, 9, 8, 107, 336, 285, 55,
    21, 71, 68, 54, 103, 104, 63, 70, 53, 52, 65, 107, 66, 108, 69, 67, 109, 105,
    338, 337, 336, 296, 285, 295, 282, 334, 293, 301, 251, 298, 333, 299, 297, 332, 284,
]
