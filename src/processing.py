import scipy.signal
import scipy.sparse
import numpy as np
import cv2
from src import config

def extract_mean_rgb(frame: np.ndarray, roi: np.ndarray) -> np.ndarray | None:
    mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pixels = frame[mask > 0]
    if len(pixels) == 0:
        return None
    return pixels.mean(axis=0)


def detrend(sig: np.ndarray, lam: float = config.DETREND_LAMBDA) -> np.ndarray:
    n = len(sig)
    H = np.eye(n)
    ones = np.ones(n)
    D = scipy.sparse.spdiags(
        np.array([ones, -2 * ones, ones]), [0, 1, 2], n - 2, n
    ).toarray()
    return (H - np.linalg.inv(H + lam ** 2 * D.T @ D)) @ sig.astype(np.float64)


def bandpass_filter(sig: np.ndarray, fps: float, lo: float, hi: float) -> np.ndarray:
    nyq = fps / 2.0
    sig64 = sig.astype(np.float64)
    if config.FILTER_TYPE == "chebyshev2":
        b, a = scipy.signal.cheby2(
            config.CHEBY_ORDER, config.CHEBY_RS, [lo / nyq, hi / nyq], btype="bandpass"
        )
    else:
        b, a = scipy.signal.butter(1, [lo / nyq, hi / nyq], btype="bandpass")
    return scipy.signal.filtfilt(b, a, sig64).astype(np.float32)


def process_bvp(rgb_buf: np.ndarray, fps: float) -> np.ndarray:
    sig = rgb_buf[:, 1].astype(np.float64)
    sig = detrend(sig)
    if len(sig) >= int(fps * 2):
        sig = bandpass_filter(sig, fps, config.CHEBY_LO, config.CHEBY_HI)
    sig -= sig.mean()
    std = sig.std()
    if std > 1e-6:
        sig /= std
    return sig.astype(np.float32)

def estimate_hr(bvp: np.ndarray, fps: float) -> float | None:
    if len(bvp) < int(fps * 2):
        return None
    freqs = np.fft.rfftfreq(len(bvp), d=1.0 / fps)
    power = np.abs(np.fft.rfft(bvp)) ** 2
    mask = (freqs >= config.HR_LO_HZ) & (freqs <= config.HR_HI_HZ)
    if not mask.any():
        return None
    return float(freqs[mask][np.argmax(power[mask])] * 60.0)
