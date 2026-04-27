# rPPG-Detection

Real-time remote photoplethysmography (rPPG) — heart rate measurement from a regular webcam by analyzing subtle color changes in facial skin caused by blood flow, without any contact sensors.

![Demo](assets/me.png)

## How it works

1. **Face detection** — MediaPipe FaceLandmarker detects 478 facial landmarks per frame
2. **ROI extraction** — three regions are masked: forehead, left cheek, right cheek
3. **Signal extraction** — mean RGB is averaged across all three ROIs per frame, buffered over a 10-second sliding window
4. **rPPG algorithm** — POS or CHROM projects the RGB signal onto a pulse-orthogonal plane to isolate the blood volume pulse (BVP)
5. **HR estimation** — FFT peak detection in the 40–180 BPM band

## Stack

| Component | Technology |
|---|---|
| Face detection | [MediaPipe FaceLandmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (Tasks API) |
| Video capture | OpenCV |
| rPPG algorithms | POS , CHROM |
| Signal processing | NumPy, SciPy (Butterworth bandpass, FFT) |


## Project structure

```
rPPG-Detection/
├── main.py                  # Entry point
├── src/
│   ├── pipeline.py          # main real-time loop
│   ├── face_detector.py     # mediapipe landmarker + ROI masking
│   ├── video_capture.py     # get_frame from camera
│   ├── processing.py        # rgb extraction, hr estimation
│   ├── visualization.py     # bvp plot, ROI overlay
│   ├── utils.py             # utils
│   └── config.py            # config
├── models/
│   ├── pos.py               # POS classic algorithm
│   └── chrom.py             # CHROM classic algorithm
└── assets/
    └── me.png
```

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

```python
RPPG_METHOD: str = "POS"   # or "CHROM"
```

## References

- Wang, W. et al. (2017). *Algorithmic Principles of Remote PPG*. IEEE TBME.
- De Haan, G. & Jeanne, V. (2013). *Robust Pulse Rate From Chrominance-Based rPPG*. IEEE TBME.
- Egorov, K. et al. (2025). *Gaze into the Heart: A Multi-View Video Dataset for rPPG*. ACM MM '25.
