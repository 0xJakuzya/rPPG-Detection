# rPPG Heart Rate Estimation

Этот проект реализует метод remote photoplethysmography (rPPG) — технологию, позволяющую оценивать частоту сердечных сокращений (Heart Rate) без физического контакта с телом.

Вместо датчиков используется обычная камера: она фиксирует едва заметные изменения цвета кожи, вызванные кровотоком. Далее эти изменения преобразуются в одномерный физиологический сигнал, из которого извлекается пульс.

## Архитектура PhysNet

![PhysNet architecture](assets/physnet_architecture.png)
Рис 1. Архитектура PhysNet + ROI-patches


## Возможности

- Извлечение multi-ROI патчей лица через MediaPipe.
- Подготовка `.npz`-окон для обучения.
- Subject-level train/validation split, чтобы пациенты не смешивались между
  train и validation.
- Обучение `baseline` CNN и `physnet`.
- Loss-функции `negpearson` и `cnn`.
- Оценка качества в HR-пространстве: MAE, RMSE, scatter plot, Bland-Altman.
- Realtime POS/CHROME baseline через веб-камеру.

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Структура

```text
rPPG-Detection/
  assets/
  data/
  models/
    baseline.py
    chrom.py
    loss.py
    physnet.py
    pos.py
  src/
    config.py
    dataset.py
    face_detector.py
    preprocessing.py
    test.py
    train.py
    utils.py
    video.py
    visualization.py
  main.py
  requirements.txt
  README.md
```
