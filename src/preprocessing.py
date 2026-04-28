import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from src import config
from src.face_detector import FaceDetector
from src.utils import extract_multi_rois_patches, normalize_patch_window, normalize_signal

def iter_rows(dataset_root: Path, camera: str, step: str):
    with (dataset_root / "db.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["camera"] != camera:
                continue
            if step != "all" and row["step"] != step:
                continue
            yield row

def fill_missing_patches(patches: np.ndarray) -> np.ndarray | None:
    flat = patches.reshape(patches.shape[0], -1)
    valid = np.isfinite(flat).all(axis=1)
    if not valid.any():
        return None
    idx = np.arange(len(flat))
    filled = flat.copy()
    for channel in range(flat.shape[1]):
        filled[:, channel] = np.interp(idx, idx[valid], flat[valid, channel])
    return filled.reshape(patches.shape).astype(np.float32)


def extract_patch_sequence(video_path: Path, detector: FaceDetector, frame_step: int = 1) -> tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(str(video_path))
    patch_values: list[np.ndarray] = []
    missing = 0
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_step != 0:
            frame_index += 1
            continue
        landmarks = detector.get_landmarks(frame)
        if landmarks is None:
            patch_values.append(
                np.full(
                    (config.MULTI_ROI_COUNT, config.ROI_PATCH_SIZE, config.ROI_PATCH_SIZE, 3),
                    np.nan,
                    dtype=np.float32,
                )
            )
            missing += 1
            frame_index += 1
            continue
        patches = extract_multi_rois_patches(detector, frame, landmarks, patch_size=config.ROI_PATCH_SIZE)
        if len(patches) != config.MULTI_ROI_COUNT:
            patch_values.append(
                np.full(
                    (config.MULTI_ROI_COUNT, config.ROI_PATCH_SIZE, config.ROI_PATCH_SIZE, 3),
                    np.nan,
                    dtype=np.float32,
                )
            )
            missing += 1
            frame_index += 1
            continue
        else:
            patch_values.append(patches.astype(np.float32))
        frame_index += 1
    cap.release()
    return np.asarray(patch_values, dtype=np.float32), missing

def save_windows(patches: np.ndarray, ppg: np.ndarray, output_dir:
                    Path, stem: str, window: int, stride: int) -> int:
    count = 0
    n = min(len(patches), len(ppg))
    patches = patches[:n]
    ppg = ppg[:n]
    output_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, n - window + 1, stride):
        end = start + window
        np.savez_compressed(
            output_dir / f"{stem}_{start:05d}.npz",
            patches=normalize_patch_window(patches[start:end]),
            ppg=normalize_signal(ppg[start:end]),
        )
        count += 1
    return count

def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector()
    processed = 0
    total_windows = 0

    try:
        for row in iter_rows(dataset_root, args.camera, args.step):
            if args.max_videos is not None and processed >= args.max_videos:
                break

            stem = Path(row["video"]).stem
            patient_id = stem.split("_", 1)[0]
            patient_dir = output_dir / patient_id
            patches, missing = extract_patch_sequence(
                dataset_root / row["video"],
                detector,
                frame_step=args.frame_step,
            )
            if len(patches) == 0:
                print(f"skip {stem}: no frames")
                continue

            missing_ratio = missing / len(patches)
            if missing_ratio > args.max_missing:
                print(f"skip {stem}: missing face ratio {missing_ratio:.2f}")
                continue

            patches = fill_missing_patches(patches)
            if patches is None:
                print(f"skip {stem}: no valid face frames")
                continue

            ppg = np.loadtxt(dataset_root / row["ppg_sync"], usecols=0).astype(np.float32)
            if args.frame_step > 1:
                ppg = ppg[::args.frame_step]
            written = save_windows(patches, ppg, patient_dir, stem, args.window, args.stride)
            processed += 1
            total_windows += written
            print(f"{stem}: {written} windows")
    finally:
        detector.close()

    print(f"processed videos: {processed}")
    print(f"saved windows: {total_windows}")
    print(f"output dir: {output_dir}")

def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multi-ROI MCD-rPPG windows.")
    parser.add_argument("--dataset-root", default="D:/mcd_rppg")
    parser.add_argument("--output-dir", default="data/mcd_rppg_windows")
    parser.add_argument("--camera", default="FullHDwebcam")
    parser.add_argument("--step", choices=["before", "after", "all"], default="all")
    parser.add_argument("--window", type=int, default=config.PHYSNET_WINDOW)
    parser.add_argument("--stride", type=int, default=150)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-missing", type=float, default=0.2)
    return parser.parse_args(args)

if __name__ == "__main__":
    main()
