![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?logo=scipy&logoColor=white)
![MediaPipe](https://img.shields.io/badge/-MediaPipe-FF6F00?logo=google&logoColor=white)

# rPPG-Detection

This project is about remote photoplethysmography (rPPG).  
It means heart rate estimation from a normal camera.  
The system looks at small color changes on the face skin.

Now the project has two parts:

- classical baseline methods: `POS` and `CHROM`
- a simple neural network on face patches: `CNN`

![Demo](assets/me.png)

## What The Project Does

The project uses `MediaPipe Face Landmarker` to find the face.
After that it takes small skin patches from the forehead and cheeks.
These patches are used for:

- classical signal methods
- preprocessing for training
- a patch-based CNN model

## Project Structure

```text
rPPG-Detection/
├── main.py
├── models/
│   ├── baseline.py
│   ├── chrom.py
│   ├── loss.py
│   └── pos.py
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── face_detector.py
│   ├── preprocessing.py
│   ├── test.py
│   ├── train.py
│   ├── utils.py
│   ├── video.py
│   └── visualization.py
└── assets/
    └── me.png
```
