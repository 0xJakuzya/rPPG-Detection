import math
import numpy as np
from scipy import signal

def pos(rgb, fs):
    win_sec = 1.6
    N = rgb.shape[0]
    H = np.zeros(N)
    l = math.ceil(win_sec * fs)
    for n in range(N):
        m = n - l
        if m >= 0:
            segment = rgb[m:n, :]
            Cn = segment / (np.mean(segment, axis=0) + 1e-9)
            Cn = Cn.T  
            S = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64) @ Cn
            std_ratio = (np.std(S[0]) / (np.std(S[1]) + 1e-9))
            h = S[0] + std_ratio * S[1]
            h -= h.mean()
            H[m:n] += h
    b, a = signal.butter(1, [0.75 / fs * 2, 3.0 / fs * 2], btype='bandpass')
    bvp = signal.filtfilt(b, a, H.astype(np.float64))
    bvp -= bvp.mean()
    bvp /= bvp.std() + 1e-9
    return bvp.astype(np.float32)

class POS:
    def __init__(self, fps: float):
        self.fps = fps

    def run(self, rgb: np.ndarray) -> np.ndarray:
        return pos(rgb, self.fps)
