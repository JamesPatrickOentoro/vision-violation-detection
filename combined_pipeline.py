import csv
import logging
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


SPEED_LIMIT_KMH = float(os.environ.get("SPEED_LIMIT_KMH", "20"))
SPEED_SMOOTHING_FRAMES = int(os.environ.get("SPEED_SMOOTHING_FRAMES", "7"))
PIXELS_PER_METER = float(os.environ.get("PIXELS_PER_METER", "9.12"))
HOMOGRAPHY_MATRIX_PATH = os.environ.get("HOMOGRAPHY_MATRIX_PATH", "homography.npy")
RANGE_THRESHOLD_METERS = float(os.environ.get("RANGE_THRESHOLD_METERS", "10"))


@dataclass
class ContraflowDetection:
    track_id: int
    bbox: Tuple[int, int, int, int]
    is_contraflow: bool
    lane: Optional[int]
    speed_kmh: float
    speed_violation: bool
    center: Tuple[int, int]
    projected_point: np.ndarray


@dataclass
class ScreenshotArtifact:
    path: Path
    frame_idx: int
    violation_type: str
    vehicle_id: str


@dataclass
class RangeViolation:
    track_id_a: int
    track_id_b: int
    distance_m: float
    point_a: Tuple[int, int]
    point_b: Tuple[int, int]


@dataclass
class CombinedArtifacts:
    video_path: Path
    csv_path: Path
    screenshots: List[ScreenshotArtifact]
    work_dir: Path


@dataclass
class ContraflowProcessingResult:
    frame_detections: Dict[int, List[ContraflowDetection]]
    events: List[Dict[str, float | int | str]]
    fps: float
    total_frames: int
    range_pairs: Dict[int, List[RangeViolation]]


logger = logging.getLogger(__name__)


class ContraflowSequentialProcessor:
    """Sequential contraflow detector used for the combined pipeline."""

    def __init__(self) -> None:
        self.detector = ContraflowDetector()
        self.device = self._select_device()
        self.model = YOLO(self._resolve_model_path())
        self.homography = self._load_homography()
        self.speed_history: Dict[int, List[np.ndarray]] = {}
        self.logged_speed_ids: set[int] = set()
        self.range_events_logged: set[Tuple[int, int]] = set()

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

    def _load_homography(self) -> Optional[np.ndarray]:
        if not HOMOGRAPHY_MATRIX_PATH:
            return None
        try:
            matrix_path = Path(HOMOGRAPHY_MATRIX_PATH)
            if matrix_path.exists():
                return np.load(str(matrix_path))
        except Exception:
            pass
        return None

    def _project_point(self, point: Tuple[float, float]) -> np.ndarray:
        src = np.array([[point]], dtype=np.float32)
        if self.homography is not None:
            try:
                dst = cv2.perspectiveTransform(src, self.homography)[0][0]
                return dst.astype(np.float32)
            except Exception:
                pass
        return src[0][0].astype(np.float32)

    def _compute_range_violations(self, detections: List[ContraflowDetection]) -> List[RangeViolation]:
        violations: List[RangeViolation] = []
        for i in range(len(detections)):
            det_a = detections[i]
            for j in range(i + 1, len(detections)):
                det_b = detections[j]
                delta = det_a.projected_point - det_b.projected_point
                distance_pixels = float(np.linalg.norm(delta))
                if PIXELS_PER_METER <= 0:
                    continue
                distance_m = distance_pixels / PIXELS_PER_METER
                if distance_m < RANGE_THRESHOLD_METERS:
                    violations.append(
                        RangeViolation(
                            track_id_a=det_a.track_id,
                            track_id_b=det_b.track_id,
                            distance_m=distance_m,
                            point_a=(det_a.center[0], det_a.center[1]),
                            point_b=(det_b.center[0], det_b.center[1]),
                        )
                    )
        return violations

    def process(self, video_path: Path) -> ContraflowProcessingResult:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for contraflow detection: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.detector.reset()

        frame_map: Dict[int, List[ContraflowDetection]] = {}
        events: List[Dict[str, float | int | str]] = []
        self.speed_history.clear()
        self.logged_speed_ids.clear()
        self.range_events_logged.clear()
        range_pairs: Dict[int, List[RangeViolation]] = {}

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

                ground_point = ((x1 + x2) / 2.0, y2)
                projected_point = self._project_point(ground_point)
                history = self.speed_history.setdefault(track_id, [])
                history.append(projected_point)
                if len(history) > SPEED_SMOOTHING_FRAMES * 5:
                    history.pop(0)

                speed_kmh = 0.0
                speed_violation = False
                if fps > 0 and len(history) > SPEED_SMOOTHING_FRAMES:
                    delta_vec = history[-1] - history[-SPEED_SMOOTHING_FRAMES]
                    distance_pixels = float(np.linalg.norm(delta_vec))
                    distance_meters = distance_pixels / PIXELS_PER_METER if PIXELS_PER_METER > 0 else 0.0
                    time_seconds = SPEED_SMOOTHING_FRAMES / fps
                    if time_seconds > 0:
                        speed_kmh = (distance_meters / time_seconds) * 3.6
                        speed_violation = speed_kmh > SPEED_LIMIT_KMH

                if speed_violation and track_id not in self.logged_speed_ids:
                    events.append(
                        {
                            "type": "speed",
                            "frame_idx": frame_idx,
                            "track_id": track_id,
                            "timestamp_ms": timestamp_ms,
                            "speed_kmh": speed_kmh,
                        }
                    )
                    self.logged_speed_ids.add(track_id)

                detections.append(
                    ContraflowDetection(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        is_contraflow=is_contraflow,
                        lane=lane,
                        speed_kmh=speed_kmh,
                        speed_violation=speed_violation,
                        center=(int(center_x), int(center_y)),
                        projected_point=projected_point,
                    )
                )

                if is_contraflow and self.detector.should_log_contraflow(track_id):
                    events.append(
                        {
                            "type": "contraflow",
                            "frame_idx": frame_idx,
                            "track_id": track_id,
                            "timestamp_ms": timestamp_ms,
                        }
                    )

            if detections:
                frame_map[frame_idx] = detections
                range_list = self._compute_range_violations(detections)
                if range_list:
                    range_pairs[frame_idx] = range_list
                    for violation in range_list:
                        pair_key = tuple(sorted((violation.track_id_a, violation.track_id_b)))
                        if pair_key not in self.range_events_logged:
                            self.range_events_logged.add(pair_key)
                            events.append(
                                {
                                    "type": "range",
                                    "frame_idx": frame_idx,
                                    "track_pair": f"{violation.track_id_a}-{violation.track_id_b}",
                                    "timestamp_ms": timestamp_ms,
                                    "distance_m": violation.distance_m,
                                }
                            )

            frame_idx += 1

        cap.release()

        return ContraflowProcessingResult(
            frame_detections=frame_map,
            events=events,
            fps=fps,
            total_frames=total_frames,
            range_pairs=range_pairs,
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
    range_pairs: Optional[List[RangeViolation]] = None,
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
        speed_label = f"{detection.speed_kmh:.1f} km/h"
        speed_color = (0, 0, 255) if detection.speed_violation else (255, 255, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        text_y = max(y1 - 15, 20)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            speed_label,
            (x1, text_y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            speed_color,
            1,
            cv2.LINE_AA,
        )

    if range_pairs:
        for violation in range_pairs:
            cv2.line(
                annotated,
                violation.point_a,
                violation.point_b,
                (0, 0, 255),
                2,
            )
            mid_x = int((violation.point_a[0] + violation.point_b[0]) / 2)
            mid_y = int((violation.point_a[1] + violation.point_b[1]) / 2)
            cv2.putText(
                annotated,
                f"{violation.distance_m:.2f} m",
                (mid_x, max(mid_y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    return annotated


def _annotate_combined_frame(
    frame: np.ndarray,
    frame_idx: int,
    stop_processor: StopVideo,
    contraflow_frames: Dict[int, List[ContraflowDetection]],
    range_frames: Dict[int, List[RangeViolation]],
) -> np.ndarray:
    annotated = _annotate_stop_detection(frame, frame_idx, stop_processor)
    contraflow_detections = contraflow_frames.get(frame_idx, [])
    range_detections = range_frames.get(frame_idx)
    annotated = _draw_contraflow_annotations(annotated, contraflow_detections, range_detections)
    return annotated


def _capture_combined_screenshot(
    video_path: Path,
    frame_idx: int,
    destination: Path,
    stop_processor: StopVideo,
    contraflow_frames: Dict[int, List[ContraflowDetection]],
    range_frames: Dict[int, List[RangeViolation]],
) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()
    cap.release()
    if not success or frame is None:
        return False

    annotated = _annotate_combined_frame(frame, frame_idx, stop_processor, contraflow_frames, range_frames)
    return cv2.imwrite(str(destination), annotated)


def _render_combined_video(
    stop_processor: StopVideo,
    contraflow_frames: Dict[int, List[ContraflowDetection]],
    range_frames: Dict[int, List[RangeViolation]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(stop_processor.video_info.fps or 30.0)
    frames_generator = sv.get_video_frames_generator(source_path=stop_processor.local_video_path)
    writer = None
    for frame_idx, frame in enumerate(frames_generator):
        annotated = _annotate_combined_frame(frame, frame_idx, stop_processor, contraflow_frames, range_frames)
        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Failed to open VideoWriter for combined output")
        writer.write(annotated)
    if writer is not None:
        writer.release()


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
    logger.info("Stop detection inference complete; running stop analysis...")
    stop_processor.analyze()
    logger.info("Stop analysis complete; starting contraflow and speed processing...")

    contraflow_processor = ContraflowSequentialProcessor()
    contraflow_result = contraflow_processor.process(local_video_path)
    logger.info("Contraflow and speed processing complete; preparing combined outputs...")

    fps = float(stop_processor.video_info.fps or contraflow_result.fps or 30.0)
    total_frames = contraflow_result.total_frames or getattr(stop_processor.video_info, "total_frames", 0) or 0

    processing_date = datetime.utcnow().date().isoformat()

    video_filename = Path(stop_processor.local_video_path).name
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[int, Tuple[str, str, str, str, str]]] = []
    screenshots: List[ScreenshotArtifact] = []

    for car_id, data in (stop_processor.analysis or {}).items():
        if data.get("Status") != "Failed to stop":
            continue

        frame_idx = stop_processor.car_left_zone.get(car_id)
        if frame_idx is None:
            continue

        timestamp_str = _format_timestamp_from_frame(frame_idx, fps, total_frames)
        rows.append((frame_idx, (processing_date, video_filename, timestamp_str, str(car_id), "stop violation")))

        screenshot_path = screenshot_dir / f"{video_name}_stop_id{car_id}_frame{frame_idx}.jpg"
        if _capture_combined_screenshot(
            local_video_path,
            frame_idx,
            screenshot_path,
            stop_processor,
            contraflow_result.frame_detections,
            contraflow_result.range_pairs,
        ):
            screenshots.append(
                ScreenshotArtifact(
                    path=screenshot_path,
                    frame_idx=frame_idx,
                    violation_type="stop violation",
                    vehicle_id=str(car_id),
                )
            )

    for event in contraflow_result.events:
        frame_idx = int(event.get("frame_idx", 0))
        event_type = event.get("type", "contraflow")
        timestamp_str = _format_timestamp_from_frame(frame_idx, fps, total_frames)

        if event_type == "speed":
            track_id = str(event.get("track_id", "-1"))
            rows.append((frame_idx, (processing_date, video_filename, timestamp_str, track_id, "speed violation")))
            screenshot_path = screenshot_dir / f"{video_name}_speed_id{track_id}_frame{frame_idx}.jpg"
            violation_label = "speed violation"
        elif event_type == "range":
            track_pair = str(event.get("track_pair", "unknown"))
            rows.append((frame_idx, (processing_date, video_filename, timestamp_str, track_pair, "range violation")))
            sanitized = track_pair.replace('/', '-').replace(':', '-')
            screenshot_path = screenshot_dir / f"{video_name}_range_{sanitized}_frame{frame_idx}.jpg"
            violation_label = "range violation"
        else:
            track_id = str(event.get("track_id", "-1"))
            rows.append((frame_idx, (processing_date, video_filename, timestamp_str, track_id, "contraflow violation")))
            screenshot_path = screenshot_dir / f"{video_name}_contraflow_id{track_id}_frame{frame_idx}.jpg"
            violation_label = "contraflow violation"

        if _capture_combined_screenshot(
            local_video_path,
            frame_idx,
            screenshot_path,
            stop_processor,
            contraflow_result.frame_detections,
            contraflow_result.range_pairs,
        ):
            screenshots.append(
                ScreenshotArtifact(
                    path=screenshot_path,
                    frame_idx=frame_idx,
                    violation_type=violation_label,
                    vehicle_id=track_pair if event_type == "range" else track_id,
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

    combined_video_path = output_dir / "inference_combined.avi"
    logger.info("Rendering combined annotated video...")
    _render_combined_video(stop_processor, contraflow_result.frame_detections, contraflow_result.range_pairs, combined_video_path)
    logger.info("Combined video rendering finished.")

    return CombinedArtifacts(
        video_path=combined_video_path,
        csv_path=csv_path,
        screenshots=screenshots,
        work_dir=output_dir,
    )
