from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib as mpl
from matplotlib import font_manager
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

BASE_DIR = Path(__file__).resolve().parent
SYNTH_DIR = (BASE_DIR / "../data/synthesized").resolve()
DATASET_DIR = (BASE_DIR / "../data/yolo11_dataset").resolve()
MODEL_NAME = "yolo11n.pt"
EPOCHS = 100
BATCH_SIZE = 64
IMG_SIZE = 640
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
RANDOM_SEED = 42

FONT_PATH = (BASE_DIR / "../fonts/NotoSansCJKtc-Regular.otf").resolve()


@dataclass
class Sample:
    image_path: Path
    ann_path: Path
    signs: List[dict]


def configure_matplotlib_fonts() -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Font file not found: {FONT_PATH}")
    font_manager.fontManager.addfont(str(FONT_PATH))
    font_prop = font_manager.FontProperties(fname=str(FONT_PATH))
    font_name = font_prop.get_name()
    mpl.rcParams["font.family"] = [font_name]
    mpl.rcParams["font.sans-serif"] = [font_name]
    mpl.rcParams["axes.unicode_minus"] = False


def discover_samples(synth_dir: Path) -> List[Sample]:
    samples: List[Sample] = []
    for img_path in synth_dir.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        ann_path = img_path.with_suffix(".json")
        if not ann_path.exists():
            continue
        with ann_path.open("r", encoding="utf-8") as ann_file:
            ann = json.load(ann_file)
        signs = ann.get("signs", [])
        samples.append(Sample(image_path=img_path, ann_path=ann_path, signs=signs))
    return samples


def build_class_map(samples: Sequence[Sample]) -> Dict[str, int]:
    categories = set()
    for sample in samples:
        for sign in sample.signs:
            label = sign.get("category")
            if label:
                categories.add(label)
    if not categories:
        raise RuntimeError("No categories detected in annotations.")
    return {name: idx for idx, name in enumerate(sorted(categories))}


def split_samples_three(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n < 3:
        raise RuntimeError("Need at least three annotated samples to perform train/val/test split.")
    train_end = max(1, int(n * train_ratio))
    val_end = train_end + max(1, int(n * val_ratio))
    if val_end >= n:
        val_end = n - 1
    train_samples = shuffled[:train_end]
    val_samples = shuffled[train_end:val_end]
    test_samples = shuffled[val_end:]
    if len(test_samples) == 0:
        val_samples, last = val_samples[:-1], val_samples[-1:]
        test_samples = list(last)
    return train_samples, val_samples, test_samples


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_label_file(label_path: Path, signs: Sequence[dict], class_map: Dict[str, int], width: int, height: int) -> None:
    lines: List[str] = []
    for sign in signs:
        label = sign.get("category")
        bbox = sign.get("bbox")
        if label not in class_map or not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        x1, x2 = sorted([float(x1), float(x2)])
        y1, y2 = sorted([float(y1), float(y2)])
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cx /= max(width, 1)
        cy /= max(height, 1)
        bw /= max(width, 1)
        bh /= max(height, 1)
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 1e-6), 1.0)
        bh = min(max(bh, 1e-6), 1.0)
        class_id = class_map[label]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def export_split(samples: Sequence[Sample], split_name: str, dataset_dir: Path, class_map: Dict[str, int]) -> None:
    img_dir = dataset_dir / split_name / "images"
    lbl_dir = dataset_dir / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        dest_img = img_dir / sample.image_path.name
        shutil.copy2(sample.image_path, dest_img)
        img = cv2.imread(str(dest_img))
        if img is None:
            raise RuntimeError(f"Failed to load copied image {dest_img}")
        label_path = lbl_dir / f"{dest_img.stem}.txt"
        write_label_file(label_path, sample.signs, class_map, img.shape[1], img.shape[0])


def write_dataset_yaml(dataset_dir: Path, class_map: Dict[str, int]) -> Path:
    yaml_path = dataset_dir / "dataset.yaml"
    names = [""] * len(class_map)
    for label, idx in class_map.items():
        names[idx] = label
    yaml_lines = [
        f"path: {dataset_dir.resolve()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "names:",
    ]
    yaml_lines.extend([f"  - {name}" for name in names])
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return yaml_path


def train_and_evaluate(data_yaml: Path, model_name: str, epochs: int, batch: int, imgsz: int) -> None:
    model = YOLO(model_name)
    model.train(data=str(data_yaml), epochs=epochs, batch=batch, imgsz=imgsz)
    model.val(data=str(data_yaml), split="val", name="val")
    model.val(data=str(data_yaml), split="test", name="test")


def main() -> None:
    synth_dir = SYNTH_DIR
    dataset_dir = DATASET_DIR
    if not synth_dir.exists():
        raise FileNotFoundError(f"Synthesized data directory not found: {synth_dir}")
    samples = discover_samples(synth_dir)
    if len(samples) < 3:
        raise RuntimeError("Need at least three annotated samples to perform train/val/test split.")
    class_map = build_class_map(samples)
    train_samples, val_samples, test_samples = split_samples_three(samples, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    reset_dir(dataset_dir)
    export_split(train_samples, "train", dataset_dir, class_map)
    export_split(val_samples, "val", dataset_dir, class_map)
    export_split(test_samples, "test", dataset_dir, class_map)
    configure_matplotlib_fonts()
    data_yaml = write_dataset_yaml(dataset_dir, class_map)
    train_and_evaluate(data_yaml, MODEL_NAME, EPOCHS, BATCH_SIZE, IMG_SIZE)


if __name__ == "__main__":
    main()
