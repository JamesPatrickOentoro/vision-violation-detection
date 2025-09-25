import cv2
from ultralytics import YOLO
from collections import defaultdict
# import supervision as sv
import numpy as np
import csv
from datetime import datetime
import os
import torch
import shutil
import logging
import subprocess
import traceback
import json
import time
import threading
from urllib.parse import unquote
from google.cloud import storage, pubsub_v1

# Import from other modules
from gpu import load_models
from check_vid import check_video_already_processed, ensure_mp4_format
from gcs import download_from_gcs, upload_results_to_gcs
from convert import scale_annotations_for_resolution

# --- CONFIGURATION ---
PROJECT_ID = "adaro-vision-poc"
SOURCE_BUCKET_NAME = "vision-poc-bucket-adaro"  # Source bucket for videos
RESULTS_BUCKET_NAME = "adaro-vision-overlay-speed"  # Bucket for results
SUBSCRIPTION_ID = "adaro-speed-detection-sub"
MAX_PROCESSED_VIDEOS = 500  # Maximum number of processed videos to keep in memory

# Output folders in results bucket
PROCESSED_VIDEOS_FOLDER = "processed-videos/"
REPORTS_FOLDER = "speed-detection-reports/"
SCREENSHOTS_FOLDER = "speed-screenshots/"

# Model paths in GCS
MODEL_GCS_PATH = "models/yolo11s.pt"  # Path to YOLO model in GCS
HOMOGRAPHY_GCS_PATH = "models/homography.npy"  # Path to homography matrix in GCS

YOLO_MODEL_FILENAME = "yolo11s.pt"
HOMOGRAPHY_FILENAME = "homography.npy"
SPEED_SMOOTHING_FRAMES = 7
SPEED_LIMIT_KMH = 20 # Set the speed limit in km/h
# Only log violations by default (set to True to log all detections)
LOG_ALL_SPEEDS = False
# --- CALIBRATION & INITIALIZATION ---
VIDEO_FPS = 15
TARGET_CLASSES = [2, 5, 7] # car, bus, truck


processing_lock = threading.Lock()  # Thread lock for safe access
global_processing_lock = threading.Lock()  # Global lock to ensure only one video processes at a time
processed_videos = set()  # Track which videos we've already processed
processing_videos = set()  # Track which videos are currently being processed
corrupted_videos = set()  # Track which videos are too corrupted to process

# This value must be calculated from your TOP-DOWN reference image
PIXELS_PER_METER = 9.12 # Example value


def format_video_timestamp(seconds: float) -> str:
    """Format a timestamp (seconds) as H:MM:SS.mmm clamped to video length."""
    seconds = max(0.0, float(seconds or 0.0))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        millis = 0
        secs += 1
        if secs == 60:
            secs = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours}:{minutes:02d}:{secs:02d}.{millis:03d}"

def main():
    """Main entry point for the worker."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Log that we're using GCE default credentials
    logging.info("🔑 Using Compute Engine default service account credentials")
    
    if torch.cuda.is_available():
        logging.info("🚀 CUDA is available - GPU acceleration enabled!")
        logging.info(f"📊 CUDA Version: {torch.version.cuda}")
        logging.info(f"🔥 PyTorch Version: {torch.__version__}")
    else:
        logging.warning("⚠️  CUDA not available - running on CPU")
    
    # Start the Pub/Sub worker
    start_worker()

def process_video(video_blob_name, model_instance, h_matrix, device):
    start_time = time.time()
    logging.info(f"🚀 Starting processing for {video_blob_name}")
    temp_dir = f"/tmp/{video_blob_name.replace('/', '_')}"
    os.makedirs(temp_dir, exist_ok=True)
    # Initialize last good frame for interpolation
    process_video.last_good_frame = None
    # Reset per-video tracking state to avoid leakage between videos
    process_video.track_history = defaultdict(list)
    process_video.display_track_history = defaultdict(list)
    process_video.violation_logged = set()
    process_video.violation_timestamps = {}

    local_input_video_path = os.path.join(temp_dir, os.path.basename(video_blob_name))
    video_name_without_ext = os.path.splitext(os.path.basename(video_blob_name))[0]
    csv_report_temp_path = os.path.join(temp_dir, f"report_{video_name_without_ext}.csv")

    # Variables to track conversion status
    input_was_converted = False
    temp_mp4_path = None
    final_output_path = None
    try:
        # Try to locate a tracker config similar to the original script
        tracker_args = {}
        try:
            possible_paths = [
                os.path.join(os.getcwd(), 'bytetrack.yaml'),
                os.path.join(os.path.dirname(__file__), 'bytetrack.yaml'),
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    tracker_args['tracker'] = p
                    logging.info(f"Using tracker config: {p}")
                    break
        except Exception:
            pass
        # Download the video from GCS
        download_start = time.time()
        download_from_gcs(video_blob_name, local_input_video_path)
        download_time = time.time() - download_start
        logging.info(f"📥 Download completed in {download_time:.2f} seconds")

        conversion_start = time.time()
        processed_input_path, input_was_converted = ensure_mp4_format(local_input_video_path, video_name_without_ext)
        conversion_time = time.time() - conversion_start
        if input_was_converted:
            logging.info(f"🔄 Video conversion completed in {conversion_time:.2f} seconds")
        
        if processed_input_path is None:
            logging.error(f"❌ Video {video_blob_name} is too heavily corrupted to process")
            logging.error(f"🚫 Creating skip report and acknowledging message")
        
            with open(csv_report_temp_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Date', 'Video Name', 'Timestamp On Video (ms)', 'Vehicle ID', 'Did overspeed'])
                csv_writer.writerow([
                    datetime.now().strftime('%Y-%m-%d'),
                    os.path.basename(video_blob_name),
                    '0:00:00.000',
                    'SKIPPED',
                    'N/A'
                ])
            # Create a simple skip indicator video
            skip_video_path = os.path.join(temp_dir, f"skipped_{video_name_without_ext}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            skip_out = cv2.VideoWriter(skip_video_path, fourcc, 25, (1280, 720))
            
            # Create a frame with skip message
            skip_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(skip_frame, "VIDEO SKIPPED", (640 - 200, 360 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(skip_frame, "SEVERELY CORRUPTED", (640 - 250, 360 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(skip_frame, "Cannot be processed", (640 - 200, 360 + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write the message frame for 3 seconds
            for _ in range(75):  # 3 seconds at 25 fps
                skip_out.write(skip_frame)
            
            skip_out.release()
            
            # Upload the skip report
            upload_results_to_gcs(RESULTS_BUCKET_NAME, skip_video_path, csv_report_temp_path, video_name_without_ext)
            
            # Add to corrupted videos set to prevent reprocessing
            with processing_lock:
                corrupted_videos.add(video_blob_name)
                logging.info(f"🚫 Added {video_blob_name} to corrupted videos list - will not process again")
            # Log timing for skipped video
            end_time = time.time()
            processing_duration = end_time - start_time
            logging.info(f"⏱️  VIDEO SKIPPED DUE TO SEVERE CORRUPTION!")
            logging.info(f"📊 Processing time: {processing_duration:.2f} seconds")
            logging.info(f"🎬 Video: {video_blob_name}")
            
            return 
        
        # Store the temporary MP4 path if conversion was done
        if input_was_converted:
            temp_mp4_path = processed_input_path


        # Open the video (either original or converted)
        cap = cv2.VideoCapture(processed_input_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {processed_input_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_value = cap.get(cv2.CAP_PROP_FPS)
        try:
            fps = int(fps_value) if fps_value and fps_value > 0 else 25
        except Exception:
            fps = 25
        if fps == 25:
            logging.warning("FPS reported as 0/invalid; falling back to 25 FPS for writer")
        
        # Reset video position to beginning for processing
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Target resolution for output video
        target_width = 1280
        target_height = 720

        # Calculate scaling factors for annotations
        scale, offset_x, offset_y = scale_annotations_for_resolution(frame_width, frame_height, target_width, target_height)
        # Create temporary output (prefer reliable codecs on headless VMs)
        use_ffmpeg_pipe = False
        ffmpeg_proc = None
        temp_mp4_output = os.path.join(temp_dir, f"processed_{video_name_without_ext}_temp.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_mp4_output, fourcc, fps, (target_width, target_height))
        if not out.isOpened():
            logging.warning("XVID codec failed, trying MP4V...")
            temp_mp4_output = os.path.join(temp_dir, f"processed_{video_name_without_ext}_temp.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_mp4_output, fourcc, fps, (target_width, target_height))
        if not out.isOpened():
            logging.warning("MP4V codec failed, trying H264...")
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(temp_mp4_output, fourcc, fps, (target_width, target_height))
        if not out.isOpened():
            # Fallback to ffmpeg rawvideo pipe if available
            try:
                import shutil as _shutil
                if _shutil.which('ffmpeg'):
                    logging.warning("OpenCV encoders failed. Falling back to ffmpeg pipe...")
                    temp_mp4_output = os.path.join(temp_dir, f"processed_{video_name_without_ext}.mp4")
                    cmd = [
                        'ffmpeg','-y',
                        '-f','rawvideo','-vcodec','rawvideo',
                        '-pixel_format','bgr24','-video_size', f'{target_width}x{target_height}',
                        '-r', str(fps), '-i','-',
                        '-an','-vcodec','libx264','-preset','veryfast','-crf','23','-movflags','+faststart',
                        temp_mp4_output
                    ]
                    ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    use_ffmpeg_pipe = True
                else:
                    raise RuntimeError("Could not open any video writer (XVID/MP4V/H264) and ffmpeg not found.")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize any video writer: {e}")
        
        logging.info(f"🎬 Processing video frames for {video_blob_name}...")
        frame_processing_start = time.time()
        violations_logged = False
        with open(csv_report_temp_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Date', 'Video Name', 'Timestamp On Video (ms)', 'Vehicle ID', 'Did overspeed'])

            total_bad_frames = 0
            total_good_frames = 0
            current_frame_index = 0
            # Get total frame count for proper termination
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                # If we can't get frame count, estimate based on duration and FPS
                duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                total_frames = int(duration * fps) if fps > 0 else 1000
                logging.warning(f"Could not get frame count, estimating {total_frames} frames based on duration and FPS")
            else:
                logging.info(f"Video has {total_frames} total frames")

            video_duration_sec = ((total_frames - 1) / fps) if fps > 0 and total_frames > 0 else None

            while current_frame_index < total_frames:
                ret, frame = cap.read()
                if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0 and np.mean(frame) > 5:
                    total_good_frames += 1
                    process_video.last_good_frame = frame.copy()
                else:
                    total_bad_frames += 1
                    if process_video.last_good_frame is not None:
                        frame = process_video.last_good_frame.copy()
                        logging.warning(f"Frame {current_frame_index} corrupted, reusing previous frame")
                    else:
                        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                        logging.warning(f"Frame {current_frame_index} corrupted and no previous frame available; using blank frame")

                if fps > 0:
                    frame_index_for_time = max(0, min(current_frame_index, total_frames - 1)) if total_frames > 0 else current_frame_index
                    frame_based_timestamp = frame_index_for_time / fps
                    if video_duration_sec is not None:
                        current_timestamp_sec = min(frame_based_timestamp, video_duration_sec)
                    else:
                        current_timestamp_sec = frame_based_timestamp
                else:
                    current_timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                # Log progress every 100 frames
                if current_frame_index % 100 == 0:
                    logging.info(
                        f"Processed {current_frame_index}/{total_frames} frames (good: {total_good_frames}, bad: {total_bad_frames}), current timestamp: {current_timestamp_sec:.2f}s"
                    )
                
                # Use YOLO tracking to get stable IDs across frames
                try:
                    if device and device.type == 'cuda':
                        results_list = model_instance.track(
                            frame, persist=True, classes=TARGET_CLASSES, verbose=False, device=device, **tracker_args
                        )
                    else:
                        results_list = model_instance.track(
                            frame, persist=True, classes=TARGET_CLASSES, verbose=False, **tracker_args
                        )
                    # Ultralytics returns a list; we need the first result
                    results = results_list[0] if isinstance(results_list, (list, tuple)) else results_list
                except Exception as e:
                    logging.warning(f"Model tracking failed on frame {current_frame_index}: {str(e)}")
                    results = None

                # Initialize tracking data structures if not already done
                if not hasattr(process_video, 'track_history'):
                    process_video.track_history = defaultdict(list)
                    process_video.display_track_history = defaultdict(list)
                    process_video.violation_logged = set()
                    process_video.violation_timestamps = {}

                annotated_frame = frame.copy()
                # Initialize empty lists for no detections case
                boxes, track_ids, class_ids = [], [], []
                
                # Only try to get detections if results is not None
                if results is not None and hasattr(results, 'boxes') and results.boxes is not None:
                    try:
                        boxes = results.boxes.xywh.cpu()
                        # Some frames may have id=None; handle gracefully
                        ids = results.boxes.id
                        track_ids = ids.int().cpu().tolist() if ids is not None else []
                        class_ids = results.boxes.cls.int().cpu().tolist() if results.boxes.cls is not None else []
                        # Ensure equal lengths
                        if len(track_ids) != len(class_ids) or len(boxes) != len(track_ids):
                            n = min(len(boxes), len(track_ids), len(class_ids))
                            boxes = boxes[:n]
                            track_ids = track_ids[:n]
                            class_ids = class_ids[:n]
                    except Exception:
                        logging.debug(f"No valid detections in frame {current_frame_index}")

                # Process each detected object
                for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                    if int(cls_id) not in [2, 5, 7]:  # Skip if not car, bus, or truck
                        continue

                    x_center, y_center, w, h = box
                    class_name = model_instance.names[cls_id]
                    
                    # Get ground point for transformation
                    ground_point = (int(x_center), int(y_center + h/2))
                    object_point_cctv = np.array([[ground_point]], dtype="float32")
                    object_point_transformed = cv2.perspectiveTransform(object_point_cctv, h_matrix)
                    transformed_point = object_point_transformed[0][0]

                    # Track history for speed calculation
                    track = process_video.track_history[track_id]
                    track.append(transformed_point)
                    if len(track) > 30:
                        track.pop(0)

                    # Calculate speed with smoothing
                    speed_kmh = 0
                    if len(track) > SPEED_SMOOTHING_FRAMES:
                        distance_pixels = np.linalg.norm(track[-1] - track[-SPEED_SMOOTHING_FRAMES])
                        distance_meters = distance_pixels / PIXELS_PER_METER
                        time_seconds = SPEED_SMOOTHING_FRAMES / fps
                        speed_kmh = (distance_meters / time_seconds) * 3.6

                    # Check for speed violations
                    is_violating = speed_kmh > SPEED_LIMIT_KMH
                    ui_color = (0, 0, 255) if is_violating else (255, 128, 0) # Red for violations, Blue for normal

                    # Log violations
                    if is_violating and track_id not in process_video.violation_logged:
                        process_video.violation_timestamps[track_id] = current_timestamp_sec
                        csv_writer.writerow([
                            datetime.now().strftime('%Y-%m-%d'),
                            os.path.basename(video_blob_name),
                            format_video_timestamp(current_timestamp_sec),
                            track_id,
                            'Yes'
                        ])
                        process_video.violation_logged.add(track_id)
                        violations_logged = True

                    # Draw annotations
                    # 1. Track tail
                    display_track = process_video.display_track_history[track_id]
                    display_track.append((int(x_center), int(y_center + h/2)))
                    if len(display_track) > 15:
                        display_track.pop(0)
                    if len(display_track) > 1:
                        cv2.polylines(annotated_frame, [np.array(display_track, dtype=np.int32)], 
                                    isClosed=False, color=ui_color, thickness=2)

                    # 2. Bounding box
                    x1, y1 = int(x_center - w/2), int(y_center - h/2)
                    x2, y2 = int(x_center + w/2), int(y_center + h/2)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), ui_color, thickness=2)

                    # 3. Speed label
                    label = f"#{track_id} {int(speed_kmh)} km/h"
                    font_scale = 0.6
                    font_thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                               font_scale, font_thickness)
                    label_x1, label_y1 = x1, y1 - text_h - 10
                    label_x2, label_y2 = x1 + text_w + 6, y1 - 10
                    cv2.rectangle(annotated_frame, (label_x1, label_y1), (label_x2, label_y2), ui_color, -1)
                    cv2.putText(annotated_frame, label, (x1 + 3, y1 - 10 - baseline), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 
                              font_thickness, cv2.LINE_AA)

                # Write the processed frame
                resized_frame = cv2.resize(annotated_frame, (target_width, target_height))
                if use_ffmpeg_pipe and ffmpeg_proc and ffmpeg_proc.stdin:
                    ffmpeg_proc.stdin.write(resized_frame.tobytes())
                else:
                    out.write(resized_frame)
                current_frame_index += 1

        frame_processing_time = time.time() - frame_processing_start
        logging.info(f"✅ Frame processing completed in {frame_processing_time:.2f} seconds")
        logging.info(f"📊 Total frames: {total_frames}, Good: {total_good_frames}, Bad: {total_bad_frames}")
        logging.info("⚠️ No speed violations detected" if not violations_logged else "🚨 Speed violations recorded")

        # Close video writer and reader
        if use_ffmpeg_pipe and ffmpeg_proc:
            try:
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait(timeout=15)
            except Exception:
                pass
        else:
            out.release()
        cap.release()

        # Verify output file size and readability before upload
        try:
            out_size = os.path.getsize(temp_mp4_output)
            logging.info(f"🗜️  Output video size: {out_size} bytes")
            test_cap = cv2.VideoCapture(temp_mp4_output)
            test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if test_cap.isOpened() else 0
            test_cap.release()
            logging.info(f"🧪 Output video frame count: {test_frames}")
            if out_size == 0 or test_frames == 0:
                logging.warning("Output video appears empty or unreadable. Consider checking codecs/ffmpeg.")
        except Exception as e:
            logging.warning(f"Could not verify output video: {e}")

        # Optionally convert AVI to MP4 for better compatibility if ffmpeg is available
        try:
            import shutil as _shutil
            if temp_mp4_output.lower().endswith('.avi') and _shutil.which('ffmpeg'):
                mp4_converted = os.path.join(temp_dir, f"processed_{video_name_without_ext}.mp4")
                cmd = [
                    'ffmpeg','-y','-i', temp_mp4_output,
                    '-c:v','libx264','-preset','veryfast','-crf','23',
                    '-c:a','aac','-movflags','+faststart', mp4_converted
                ]
                logging.info("Converting AVI to MP4 via ffmpeg for compatibility...")
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Replace output path to upload MP4
                temp_mp4_output = mp4_converted
        except Exception as e:
            logging.warning(f"FFmpeg conversion to MP4 failed: {e}")

        # Upload results to GCS with retries (honor return status)
        max_upload_retries = 3
        for attempt in range(max_upload_retries):
            success = upload_results_to_gcs(
                RESULTS_BUCKET_NAME, temp_mp4_output, csv_report_temp_path, video_name_without_ext
            )
            if success:
                break
            if attempt == max_upload_retries - 1:
                raise RuntimeError("Failed to upload results after retries.")
            logging.warning(f"Upload attempt {attempt + 1} failed. Retrying...")
    except Exception as e:
        logging.error(f"Error during processing of video {video_blob_name}: {e}")
        traceback.print_exc()
        raise
    finally:
        # Clean up temporary files
        try:
            if temp_mp4_path and os.path.exists(temp_mp4_path):
                logging.info(f"Cleaning up temporary MP4 file: {temp_mp4_path}")
                os.unlink(temp_mp4_path)
        except Exception as e:
            logging.warning(f"Could not clean up temporary MP4 file: {e}")
        
        logging.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

def start_worker():
    """Start the Pub/Sub worker."""
    logging.info("--- Starting Pub/Sub Pull Worker ---")
    # Load models
    model, H, device = load_models()

    # Initialize subscriber client without explicit credentials
    try:
        # First try to get GCE credentials
        from google.auth import compute_engine, default
        try:
            credentials = compute_engine.Credentials()
            logging.info("Using Compute Engine credentials")
        except Exception:
            # Fall back to default credentials
            credentials, project = default()
            logging.info("Using default credentials")
            
        subscriber = pubsub_v1.SubscriberClient(credentials=credentials)
        subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
        logging.info(f"Successfully initialized Pub/Sub client")
    except Exception as e:
        logging.error(f"Failed to initialize credentials: {e}")
        raise

    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        """
        This function is executed for each message pulled from the subscription.
        """
        try:
            logging.info(f"Received message ID: {message.message_id}")
            # The message data is the GCS event notification as a JSON string
            data_str = message.data.decode("utf-8")
            data = json.loads(data_str)
            
            # Some publishers may URL-encode object names; defensively decode
            video_blob_name = unquote(data.get("name", ""))
            bucket_name = data.get("bucket")

            if not video_blob_name or not bucket_name:
                logging.warning(f"Invalid message format, missing 'name' or 'bucket'. Skipping.")
                message.ack() # Acknowledge invalid messages to remove them
                return

            logging.info(f"Processing event for file: {video_blob_name} in bucket: {bucket_name}")
            logging.debug(f"Raw values -> name: {repr(video_blob_name)}, bucket: {repr(bucket_name)}")

            # Process all videos from the source bucket
            if bucket_name != SOURCE_BUCKET_NAME:
                logging.info(f"File is not in the source bucket {SOURCE_BUCKET_NAME}. Skipping.")
                message.ack()
                return
            
            # Skip directory paths (folders)
            if video_blob_name.endswith('/'):
                logging.info(f"File {video_blob_name} is a directory path. Skipping.")
                message.ack()
                return
            
            # Skip non-video files (like .txt placeholder files)
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
            file_extension = os.path.splitext(video_blob_name.lower())[1]
            if file_extension not in video_extensions:
                logging.info(f"File {video_blob_name} is not a video file (extension: {file_extension}). Skipping.")
                message.ack()
                return

            # Global processing lock to ensure only one video processes at a time
            with global_processing_lock:
                logging.info(f"🔒 Acquired global processing lock for {video_blob_name}")
                
                # Thread-safe duplicate detection - MUST be atomic
                should_process = False
                with processing_lock:
                    # Check if already processed
                    if video_blob_name in processed_videos:
                        logging.info(f"Video {video_blob_name} already processed (in-memory). Skipping duplicate.")
                        message.ack()
                        return
                    
                    # Check if currently being processed
                    if video_blob_name in processing_videos:
                        logging.info(f"Video {video_blob_name} is currently being processed. Skipping duplicate.")
                        message.ack()
                        return
                    
                    # Check if video is known to be corrupted
                    if video_blob_name in corrupted_videos:
                        logging.info(f"Video {video_blob_name} is known to be corrupted. Skipping to prevent reprocessing.")
                        message.ack()
                        return
                    
                    # Add to processing set to prevent other threads from processing same video
                    processing_videos.add(video_blob_name)
                    should_process = True
                    logging.info(f"🎬 Processing video {video_blob_name} for the first time.")
                
                # Only proceed if we should process (outside the lock)
                if not should_process:
                    return

                # Check if video was already processed (GCS check) - outside lock for performance
                if check_video_already_processed(video_blob_name):
                    logging.info(f"Video {video_blob_name} already has results in GCS. Skipping.")
                    with processing_lock:
                        processing_videos.discard(video_blob_name)
                        processed_videos.add(video_blob_name)
                    message.ack()
                    return

                # Note: Cross-instance duplicate prevention removed - using in-memory tracking only

                # Process the video
                try:
                    logging.info(f"🚀 Starting video processing: {video_blob_name}")
                    process_video(video_blob_name, model, H, device)
                    
                    # Move from processing to processed
                    with processing_lock:
                        processing_videos.discard(video_blob_name)
                        processed_videos.add(video_blob_name)
                        
                        # Clean up old entries to prevent memory issues
                        if len(processed_videos) > 500:
                            # Remove oldest entries keeping only last 500
                            old_videos = list(processed_videos)[:-500]
                            for old_video in old_videos:
                                processed_videos.discard(old_video)
                            logging.info(f"Cleaned up {len(old_videos)} old video entries from memory.")

                        # Also clean up corrupted videos set
                        if len(corrupted_videos) > MAX_PROCESSED_VIDEOS:
                            old_corrupted = list(corrupted_videos)[:-500]
                            for old_video in old_corrupted:
                                corrupted_videos.discard(old_video)
                            logging.info(f"Cleaned up {len(old_corrupted)} old corrupted video entries from memory.")
                    
                    # Acknowledge the message so it's not sent again
                    logging.info(f"✅ Successfully processed {video_blob_name}. Acknowledging message.")
                    message.ack()
                    
                    logging.info(f"📊 Total videos processed in this session: {len(processed_videos)}")
                    logging.info(f"🚫 Total corrupted videos skipped: {len(corrupted_videos)}")
                    
                except Exception as e:
                    logging.error(f"❌ Failed to process {video_blob_name}: {e}")
                    # Remove from processing set on failure
                    with processing_lock:
                        processing_videos.discard(video_blob_name)
                    # Don't acknowledge the message so it can be retried
                    message.nack()
                
                logging.info(f"🔓 Released global processing lock for {video_blob_name}")

        except Exception as e:
            logging.error(f"CRITICAL ERROR processing message: {e}")
            traceback.print_exc()
            # Do not acknowledge the message, so Pub/Sub will try to resend it later.
            message.nack()

    # The subscriber pulls messages in the background.
    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    logging.info(f"Listening for messages on {subscription_path}...")

    # Keep the main thread alive to allow the background subscriber to work.
    try:
        # You can adjust the timeout. None means it will wait indefinitely.
        streaming_pull_future.result()
    except TimeoutError:
        streaming_pull_future.cancel()
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        streaming_pull_future.result()

if __name__ == "__main__":
    main()
