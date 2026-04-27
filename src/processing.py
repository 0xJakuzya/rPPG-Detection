import scipy.signal
import numpy as np
import cv2

def extract_mean_rgb(frame: np.ndarray, roi: np.ndarray) -> np.ndarray | None:
    mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    pixels = frame[mask > 0]
    if len(pixels) == 0:
        return None
    return pixels.mean(axis=0)

def estimate_hr(bvp: np.ndarray, fps: float) -> float | None:
    if len(bvp) < int(fps * 2):
        return None
    freqs = np.fft.rfftfreq(len(bvp), d=1.0 / fps)
    power = np.abs(np.fft.rfft(bvp)) ** 2
    mask = (freqs >= 0.67) & (freqs <= 3.0)
    if not mask.any():
        return None
    return float(freqs[mask][np.argmax(power[mask])] * 60.0)

# def filter_signal(signal, rate, freq, mode='high', order=4):
#     hb_n_freq = freq / (rate / 2)
#     b, a = scipy.signal.butter(order, hb_n_freq, mode)
#     filtered = scipy.signal.filtfilt(b, a, signal)
#     filtered = filtered.astype(signal.dtype)
#     return filtered

# def bandpass_filter(signal, rate, low_freq, high_freq, order=4):
#     signal = filter_signal(signal, rate, high_freq, mode='low',  order=order)
#     signal = filter_signal(signal, rate, low_freq,  mode='high', order=order)
#     return signal

# def fft_hr(sig: np.ndarray, fps: float, lo: float = 0.5, hi: float = 3.5) -> float:
#     n = 2 ** (len(sig) - 1).bit_length()
#     f, pxx = scipy_signal.periodogram(sig, fs=fps, nfft=n, detrend=False)
#     mask = (f >= lo) & (f <= hi)
#     return float(f[mask][np.argmax(pxx[mask])] * 60)

# def load_ppg_sync(path: str) -> tuple[np.ndarray, float]:
#     data = np.loadtxt(path)
#     vals = data[:, 0].astype(np.float32)
#     total_time = data[:, 1].sum()
#     fps = len(vals) / total_time if total_time > 0 else 100.0
#     return vals, fps

# def video_to_rgb(video):
#     mask = video != 0
#     rgb = np.zeros((video.shape[0], 3), dtype='float32')
#     for i in range(video.shape[0]):
#         rgb[i, 0] = video[i, :, :, 0][mask[i, :, :, 0]].mean()
#         rgb[i, 1] = video[i, :, :, 1][mask[i, :, :, 1]].mean()
#         rgb[i, 2] = video[i, :, :, 2][mask[i, :, :, 2]].mean()
#     return rgb

# def resample_ppg(ppg: np.ndarray, ppg_fps: float, video_fps: float, n_frames: int) -> np.ndarray:
#     resampled = scipy_signal.resample(ppg, int(len(ppg) / ppg_fps * video_fps))
#     if len(resampled) >= n_frames:
#         return resampled[:n_frames].astype(np.float32)
#     pad = np.zeros(n_frames - len(resampled), dtype=np.float32)
#     return np.concatenate([resampled, pad]).astype(np.float32)

