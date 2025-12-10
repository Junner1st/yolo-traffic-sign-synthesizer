from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import subprocess

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = (BASE_DIR / "../data/videos/sample-day-1.mp4").resolve()
MODEL_PATH = (BASE_DIR / "../src/runs/detect/train/weights/best.pt").resolve()
OUTPUT_PATH = (BASE_DIR / "../data/recognized/sample_recognized_tracked.mp4").resolve()
SAVE_VIDEO = True
MAX_FRAMES = None
CHUNK_SIZE = 32
WORKERS = 1
CONFIDENCE = 0.25
IOU = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class RecognitionStats:
    frame_count: int
    detection_time: float
    detections_total: int

    @property
    def fps(self) -> float:
        return 0.0 if self.detection_time == 0 else self.frame_count / self.detection_time

    @property
    def avg_time_per_frame(self) -> float:
        return 0.0 if self.frame_count == 0 else self.detection_time / self.frame_count


class TemporalTracker:
    def __init__(self, iou_thresh: float = 0.4, min_frames: int = 1, max_missing: int = 5):
        self.tracks: List[dict] = []
        self.next_id = 0
        self.iou_thresh = iou_thresh
        self.min_frames = min_frames
        self.max_missing = max_missing

    def _iou(self, a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
        return inter / (area_a + area_b - inter)

    def update(self, boxes, scores, classes):
        updated_tracks: List[dict] = []
        used = set()

        for track in self.tracks:
            best_iou = 0.0
            best_idx = -1
            for i, box in enumerate(boxes):
                if i in used:
                    continue
                if track["cls"] != classes[i]:
                    continue
                iou = self._iou(track["bbox"], box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= self.iou_thresh and best_idx >= 0:
                i = best_idx
                used.add(i)
                track["bbox"] = boxes[i]
                track["score"] = 0.7 * track["score"] + 0.3 * scores[i]
                track["frames"] += 1
                track["missing"] = 0
                updated_tracks.append(track)
            else:
                track["missing"] += 1
                if track["missing"] <= self.max_missing:
                    updated_tracks.append(track)

        for i, box in enumerate(boxes):
            if i in used:
                continue
            track = {
                "id": self.next_id,
                "bbox": box,
                "score": float(scores[i]),
                "cls": int(classes[i]),
                "frames": 1,
                "missing": 0,
            }
            self.next_id += 1
            updated_tracks.append(track)

        self.tracks = updated_tracks
        visible = [t for t in self.tracks if t["frames"] >= self.min_frames and t["missing"] == 0]
        return visible


def ensure_paths(video_path: Path, model_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")


def build_model_pool(model_path: Path, workers: int) -> Tuple[queue.SimpleQueue, List[str]]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    models = [YOLO(str(model_path)) for _ in range(workers)]
    model_queue: queue.SimpleQueue = queue.SimpleQueue()
    for model in models:
        model_queue.put(model)
    names = models[0].model.names if hasattr(models[0], "model") else {}
    if isinstance(names, dict):
        ordered = [names[idx] for idx in range(len(names))]
    elif isinstance(names, (list, tuple)):
        ordered = list(names)
    else:
        ordered = [str(names)]
    return model_queue, ordered


def _predict_frame(
    frame,
    model_queue: queue.SimpleQueue,
    conf: float,
    iou: float,
):
    model = model_queue.get()
    try:
        results = model.predict(frame, device="cpu", conf=conf, iou=iou, verbose=False)
        return results[0]
    finally:
        model_queue.put(model)


def annotate_frame(frame, prediction, names: Sequence[str], tracks: Sequence[dict] | None = None):
    annotated = frame.copy()

    if tracks:
        for track in tracks:
            x1, y1, x2, y2 = map(int, track["bbox"])
            label = names[track["cls"]] if 0 <= track["cls"] < len(names) else str(track["cls"])
            text = f"{label}#{track['id']} {track['score']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(annotated, text, (x1, max(0, y1 - 6)), FONT, 0.5, (0, 0, 0), 2)
            cv2.putText(annotated, text, (x1, max(0, y1 - 6)), FONT, 0.5, (255, 255, 255), 1)
        return annotated
    return annotated


def _extract_prediction_arrays(prediction):
    if prediction is None or prediction.boxes is None or len(prediction.boxes) == 0:
        return [], [], []
    boxes = prediction.boxes.xyxy.cpu().numpy()
    scores = prediction.boxes.conf.cpu().numpy()
    classes = prediction.boxes.cls.cpu().numpy().astype(int)
    box_list = [tuple(map(float, bbox)) for bbox in boxes]
    score_list = [float(s) for s in scores]
    class_list = [int(c) for c in classes]
    return box_list, score_list, class_list


def run_inference(
    frames: Sequence,
    model_queue: queue.SimpleQueue,
    workers: int,
    conf: float,
    iou: float,
) -> List:
    if workers <= 1:
        return [_predict_frame(frame, model_queue, conf, iou) for frame in frames]
    predictions: List = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_predict_frame, frame, model_queue, conf, iou) for frame in frames]
        for future in futures:
            predictions.append(future.result())
    return predictions


def process_video(
    video_path: Path,
    model_path: Path,
    output_path: Path,
    save_video: bool,
    max_frames: int | None,
    chunk_size: int,
    workers: int,
    conf: float,
    iou: float,
) -> RecognitionStats:
    ensure_paths(video_path, model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-2:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ffmpeg_proc = None
    if save_video:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd_str = f"ffmpeg -y -f rawvideo -vcodec rawvideo -pix_fmt bgr24 -s {width}x{height} -r {fps} -i - -an -vcodec libx264 -preset veryfast -crf 23 {output_path}"
        ffmpeg_proc = subprocess.Popen(cmd_str, stdin=subprocess.PIPE, shell=True)

    model_queue, names = build_model_pool(model_path, workers)
    tracker = TemporalTracker()

    total_frames = 0
    total_detections = 0
    total_detection_time = 0.0
    chunk: List = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            chunk.append(frame)
            total_frames += 1
            if max_frames is not None and total_frames >= max_frames:
                break
            if len(chunk) >= chunk_size:
                elapsed, predictions = _process_chunk(
                    chunk, model_queue, ffmpeg_proc, names, conf, iou, workers, tracker
                )
                total_detection_time += elapsed
                total_detections += _count_detections(predictions)
                chunk = []
        if chunk:
            elapsed, predictions = _process_chunk(
                chunk, model_queue, ffmpeg_proc, names, conf, iou, workers, tracker
            )
            total_detection_time += elapsed
            total_detections += _count_detections(predictions)
    finally:
        cap.release()
        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

    return RecognitionStats(frame_count=total_frames, detection_time=total_detection_time, detections_total=total_detections)


def _process_chunk(
    frames: Sequence,
    model_queue: queue.SimpleQueue,
    ffmpeg_proc,
    names: Sequence[str],
    conf: float,
    iou: float,
    workers: int,
    tracker: TemporalTracker,
) -> Tuple[float, List]:
    start = time.perf_counter()
    predictions = run_inference(frames, model_queue, workers, conf, iou)
    elapsed = time.perf_counter() - start
    for frame, prediction in zip(frames, predictions):
        boxes, scores, classes = _extract_prediction_arrays(prediction)
        tracks = tracker.update(boxes, scores, classes)
        if ffmpeg_proc is not None:
            annotated = annotate_frame(frame, prediction, names, tracks)
            ffmpeg_proc.stdin.write(annotated.tobytes())
    return elapsed, predictions


def _count_detections(predictions: Sequence) -> int:
    total = 0
    for item in predictions:
        if hasattr(item, "boxes") and item.boxes is not None:
            total += len(item.boxes)
    return total


def main() -> None:
    stats = process_video(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        save_video=SAVE_VIDEO,
        max_frames=MAX_FRAMES,
        chunk_size=CHUNK_SIZE,
        workers=WORKERS,
        conf=CONFIDENCE,
        iou=IOU,
    )
    print("=== Recognition Summary ===")
    print(f"Frames processed: {stats.frame_count}")
    print(f"Total detection time (excl. model load): {stats.detection_time:.3f}s")
    print(f"Average per frame: {stats.avg_time_per_frame:.4f}s")
    print(f"Throughput: {stats.fps:.2f} FPS")
    print(f"Detections counted: {stats.detections_total}")


if __name__ == "__main__":
    main()
