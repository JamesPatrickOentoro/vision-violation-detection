import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

try:
    import torch
except ImportError:  # Torch may be unavailable in some environments
    torch = None

from contraflow.config import (
    DEFAULT_MODEL_SIZE,
    LANE_COLORS,
    MODEL_CLASSES,
    PYTORCH_MODEL_PATHS,
    ROIS,
    TENSORRT_MODEL_PATHS,
)
from contraflow.detector import ContraflowDetector
from stopvideo import StopVideo


@dataclass
class ContraflowDetection:
    track_id: int
    bbox: Tuple[int, int, int, int]
    is_contraflow: bool
    lane: Optional[int]


@dataclass
class ContraflowProcessingResult:
    frame_detections: Dict[int, List[ContraflowDetection]]
    events: List[Dict[str, int]]
    fps: float
    total_frames: int


@dataclass
class ScreenshotArtifact:
    path: Path
    frame_idx: int
    violation_type: str
    vehicle_id: int


@dataclass
class CombinedArtifacts:
    video_path: Path
    csv_path: Path
    screenshots: List[ScreenshotArtifact]


class ContraflowSequentialProcessor:
    """Sequential contraflow detector used for the combined pipeline."""

    def __init__(self) -> None:
        self.detector = ContraflowDetector()
        self.device = self._select_device()
        self.model = YOLO(self._resolve_model_path())

    def _select_device(self) -> str:
        if torch is not None and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _resolve_model_path(self) -> str:
        model_path = TENSORRT_MODEL_PATHS.get(DEFAULT_MODEL_SIZE)
        if model_path and os.path.exists(model_path):
            return model_path

        fallback_path = PYTORCH_MODEL_PATHS.get(DEFAULT_MODEL_SIZE)
        if fallback_path and os.path.exists(fallback_path):
            return fallback_path

        # Final fallback lets Ultralytics handle model download if needed
        return "yolo11n.pt"

    def process(self, video_path: Path) -> ContraflowProcessingResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for contraflow detection: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.detector.reset()

        frame_map: Dict[int, List[ContraflowDetection]] = {}
        events: List[Dict[str, int]] = []

        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) or 0)
            results = self.model.track(
                frame,
                persist=True,
                classes=MODEL_CLASSES,
                verbose=False,
                device=self.device,
            )

            if not results:
                frame_idx += 1
                continue

            result = results[0]
            boxes = getattr(result.boxes, "xyxy", None)
            track_ids = getattr(result.boxes, "id", None)

            if boxes is None or track_ids is None:
                frame_idx += 1
                continue

            ids = track_ids.int().cpu().tolist()
            boxes_np = boxes.cpu().numpy()

            detections: List[ContraflowDetection] = []
            for box, track_id in zip(boxes_np, ids):
                x1, y1, x2, y2 = [int(v) for v in box]
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                center_point = (center_x, center_y)

                self.detector.update_track_history(track_id, center_point)
                is_contraflow = self.detector.is_contraflow(track_id, center_point)
                lane = self.detector.vehicle_assigned_lane.get(track_id)

                detections.append(
                    ContraflowDetection(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        is_contraflow=is_contraflow,
                        lane=lane,
                    )
                )

                if is_contraflow and self.detector.should_log_contraflow(track_id):
                    events.append(
                        {
                            "frame_idx": frame_idx,
                            "track_id": track_id,
                            "timestamp_ms": timestamp_ms,
                        }
                    )

            if detections:
                frame_map[frame_idx] = detections

            frame_idx += 1

        cap.release()

        return ContraflowProcessingResult(
            frame_detections=frame_map,
            events=events,
            fps=fps,
            total_frames=total_frames,
        )


def _format_timestamp_from_frame(frame_idx: int, fps: float, total_frames: int) -> str:
    if fps <= 0:
        fps = 30.0

    if total_frames > 0:
        frame_idx = max(0, min(frame_idx, total_frames - 1))
    else:
        frame_idx = max(0, frame_idx)

    total_millis = int(round((frame_idx / fps) * 1000))
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _annotate_stop_detection(frame: np.ndarray, frame_idx: int, processor: StopVideo) -> np.ndarray:
    annotated = frame.copy()
    annotated = processor.stopzone_annotator.annotate(scene=annotated)
    annotated = processor.outzone_annotator.annotate(scene=annotated)
    annotated = processor.road_annotator.annotate(scene=annotated)

    point_x, point_y = processor.point
    cv2.circle(annotated, (int(point_x), int(point_y)), 10, (0, 255, 255), -1)

    if frame_idx in processor.car_detections:
        car_detections = processor.car_detections[frame_idx]
        car_id = processor.get_ids(car_detections)

        if car_id in processor.analysis:
            car_center = processor.get_car_center(frame_idx)
            if car_center:
                cv2.circle(annotated, (int(car_center[0]), int(car_center[1])), 10, (0, 255, 0), -1)
                cv2.line(
                    annotated,
                    (int(point_x), int(point_y)),
                    (int(car_center[0]), int(car_center[1])),
                    (0, 255, 0),
                    2,
                )

            label = processor.add_label(frame_idx)
            color = processor.get_color(label)

            if color == "green":
                annotated = processor.label_annotator_green.annotate(annotated, car_detections, labels=label)
                annotated = processor.box_annotator_green.annotate(annotated, car_detections)
            elif color == "red":
                annotated = processor.label_annotator_red.annotate(annotated, car_detections, labels=label)
                annotated = processor.box_annotator_red.annotate(annotated, car_detections)
            else:
                annotated = processor.label_annotator_gray.annotate(annotated, car_detections, labels=label)
                annotated = processor.box_annotator_gray.annotate(annotated, car_detections)

    return annotated


def _draw_contraflow_annotations(
    frame: np.ndarray,
    detections: List[ContraflowDetection],
) -> np.ndarray:
    annotated = frame.copy()

    for lane_id, roi in ROIS.items():
        color = LANE_COLORS.get(lane_id, (255, 255, 255))
        cv2.polylines(annotated, [roi], True, color, 2)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        if detection.is_contraflow:
            box_color = (0, 0, 255)
            status = "Contraflow"
        else:
            box_color = (0, 180, 255)
            status = "OK"

        lane_text = f"L{detection.lane}" if detection.lane else "L?"
        label = f"#{detection.track_id} {lane_text} {status}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def _annotate_combined_frame(
    frame: np.ndarray,
    frame_idx: int,
    stop_processor: StopVideo,
    contraflow_frames: Dict[int, List[ContraflowDetection]],
) -> np.ndarray:
    annotated = _annotate_stop_detection(frame, frame_idx, stop_processor)
    contraflow_detections = contraflow_frames.get(frame_idx, [])
    if contraflow_detections:
        annotated = _draw_contraflow_annotations(annotated, contraflow_detections)
    else:
        # Ensure lane overlays are still visible even without detections
        annotated = _draw_contraflow_annotations(annotated, [])
    return annotated


def _capture_combined_screenshot(
    video_path: Path,
    frame_idx: int,
    destination: Path,
    stop_processor: StopVideo,
    contraflow_frames: Dict[int, List[ContraflowDetection]],
) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    cap.release()
    if not success or frame is None:
        return False

    annotated = _annotate_combined_frame(frame, frame_idx, stop_processor, contraflow_frames)
    return cv2.imwrite(str(destination), annotated)


def _render_combined_video(
    stop_processor: StopVideo,
    contraflow_frames: Dict[int, List[ContraflowDetection]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_info = stop_processor.video_info

    with sv.VideoSink(target_path=str(output_path), video_info=video_info) as sink:
        frames_generator = sv.get_video_frames_generator(source_path=stop_processor.local_video_path)
        for frame_idx, frame in enumerate(frames_generator):
            annotated = _annotate_combined_frame(frame, frame_idx, stop_processor, contraflow_frames)
            sink.write_frame(annotated)


def process_video_combined(
    video_name: str,
    local_video_path: Path,
    wheel_model_path: str,
    vehicle_model_path: str,
    screenshot_dir: Path,
) -> CombinedArtifacts:
    stop_processor = StopVideo(
        video_name=video_name,
        local_video_path=str(local_video_path),
        wheel_model_path=wheel_model_path,
        vehicle_model_path=vehicle_model_path,
    )

    stop_processor.inference()
    stop_processor.analyze()

    contraflow_processor = ContraflowSequentialProcessor()
    contraflow_result = contraflow_processor.process(local_video_path)

    fps = float(stop_processor.video_info.fps or contraflow_result.fps or 30.0)
    total_frames = contraflow_result.total_frames or getattr(stop_processor.video_info, "total_frames", 0) or 0

    processing_date = datetime.utcnow().date().isoformat()

    video_filename = Path(stop_processor.local_video_path).name
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[int, Tuple[str, str, str, int, str]]] = []
    screenshots: List[ScreenshotArtifact] = []

    for car_id, data in (stop_processor.analysis or {}).items():
        if data.get("Status") != "Failed to stop":
            continue

        frame_idx = stop_processor.car_left_zone.get(car_id)
        if frame_idx is None:
            continue

        timestamp_str = _format_timestamp_from_frame(frame_idx, fps, total_frames)
        rows.append((frame_idx, (processing_date, video_filename, timestamp_str, car_id, "stop violation")))

        screenshot_path = screenshot_dir / f"{video_name}_stop_id{car_id}_frame{frame_idx}.jpg"
        if _capture_combined_screenshot(
            local_video_path,
            frame_idx,
            screenshot_path,
            stop_processor,
            contraflow_result.frame_detections,
        ):
            screenshots.append(
                ScreenshotArtifact(
                    path=screenshot_path,
                    frame_idx=frame_idx,
                    violation_type="stop violation",
                    vehicle_id=car_id,
                )
            )

    for event in contraflow_result.events:
        frame_idx = event["frame_idx"]
        track_id = event["track_id"]
        timestamp_str = _format_timestamp_from_frame(frame_idx, fps, total_frames)
        rows.append((frame_idx, (processing_date, video_filename, timestamp_str, track_id, "contraflow violation")))

        screenshot_path = screenshot_dir / f"{video_name}_contraflow_id{track_id}_frame{frame_idx}.jpg"
        if _capture_combined_screenshot(
            local_video_path,
            frame_idx,
            screenshot_path,
            stop_processor,
            contraflow_result.frame_detections,
        ):
            screenshots.append(
                ScreenshotArtifact(
                    path=screenshot_path,
                    frame_idx=frame_idx,
                    violation_type="contraflow violation",
                    vehicle_id=track_id,
                )
            )

    output_dir = Path(stop_processor.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{video_name}_vision_detection_report.csv"
    rows.sort(key=lambda item: item[0])

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Video Name", "Timestamp On Video (ms)", "Vehicle ID", "Did violate"])
        for _, row in rows:
            writer.writerow(row)

    combined_video_path = output_dir / "inference_combined.mp4"
    _render_combined_video(stop_processor, contraflow_result.frame_detections, combined_video_path)

    return CombinedArtifacts(
        video_path=combined_video_path,
        csv_path=csv_path,
        screenshots=screenshots,
    )
