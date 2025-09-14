import os
import csv
import json
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional

import cv2
from google.api_core import exceptions as gax_exceptions
from google.cloud import pubsub_v1
from google.cloud import storage

from stopvideo import StopVideo


# Configuration via environment variables with sensible defaults
PROJECT_ID = os.environ.get("PROJECT_ID", "adaro-vision-poc")
SUBSCRIPTION_ID = os.environ.get("SUBSCRIPTION_ID", "adaro-stop-detection-sub")
# Local model paths
WHEEL_MODEL_PATH = os.environ.get("WHEEL_MODEL_PATH", "yolo11n.pt")
VEHICLE_MODEL_PATH = os.environ.get("VEHICLE_MODEL_PATH", "yolo11s.pt")

# Model source in GCS
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "adaro-vision-stop-detection")
WHEEL_MODEL_GCS_OBJECT = os.environ.get("WHEEL_MODEL_GCS_OBJECT", "yolo11n.pt")
VEHICLE_MODEL_GCS_OBJECT = os.environ.get("VEHICLE_MODEL_GCS_OBJECT", "yolo11s.pt")

# Destination bucket/prefix for processed videos
DEST_BUCKET = os.environ.get("DEST_BUCKET", "adaro-vision-stop-detection")
DEST_PREFIX = os.environ.get("DEST_PREFIX", "processed-videos/")

# Directory to place downloaded videos
DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_DIR", "downloads"))
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def convert_avi_to_mp4(input_path: str, output_path: str) -> bool:
    """Convert AVI video to MP4 format using FFmpeg (improved reliability)"""
    try:
        # Check if FFmpeg is available
        if shutil.which('ffmpeg') is not None:
            logging.info("Converting AVI to MP4 using FFmpeg...")

            # Robust re-encode with corruption handling
            cmd = [
                "ffmpeg",
                "-y",                          # Overwrite output file without asking
                "-err_detect", "ignore_err",   # Ignore errors and continue
                "-fflags", "+genpts",          # Generate presentation timestamps
                "-avoid_negative_ts", "make_zero",  # Handle timestamp issues
                "-i", input_path,
                "-vf", "scale=1280:720",        # Resize to 1280x720
                "-b:v", "572k",                 # Set video bitrate to 572kbps
                "-c:v", "libx264",             # Use H.264 codec
                "-c:a", "aac",                 # Convert audio to AAC
                "-movflags", "+faststart",     # Optional: faster playback start
                "-max_muxing_queue_size", "1024",  # Handle large queues
                output_path
            ]

            logging.info(f"Running FFmpeg command: {' '.join(cmd)}")

            # Run FFmpeg with timeout and capture output
            subprocess.run(
                cmd,
                check=True,
                timeout=300,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logging.info("FFmpeg conversion successful")

            # Verify the converted file has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logging.info(f"Converted MP4 file size: {file_size} bytes")

                # Additional validation: check if video can be opened
                test_cap = cv2.VideoCapture(output_path)
                if test_cap.isOpened():
                    frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    test_cap.release()
                    if frame_count > 0:
                        logging.info(f"Converted MP4 verified: {frame_count} frames")
                        return True
                    else:
                        logging.error("Converted MP4 has 0 frames")
                        return False
                else:
                    logging.error("Converted MP4 cannot be opened")
                    return False
            else:
                logging.error("Converted MP4 file is empty or missing")
                return False
        else:
            logging.error("FFmpeg not found! Please install FFmpeg: sudo apt-get install -y ffmpeg")
            return False

    except subprocess.TimeoutExpired:
        logging.error("FFmpeg conversion timed out after 5 minutes")
        return False
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
        logging.error(f"FFmpeg failed with return code {e.returncode}")
        if err:
            logging.error(f"FFmpeg stderr: {err}")
        return False
    except Exception as e:
        logging.error(f"Error during FFmpeg conversion: {e}")
        return False


def download_gcs_object(bucket_name: str, object_name: str, dest_path: Path) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest_path))


def is_video_path(path: str) -> bool:
    ext = Path(path).suffix.lower()
    return ext in {".mp4", ".avi", ".mov", ".mkv"}


def process_video(local_video_path: Path) -> tuple[Path, Path | None]:
    video_name = local_video_path.stem
    processor = StopVideo(
        video_name=video_name,
        local_video_path=str(local_video_path),
        wheel_model_path=WHEEL_MODEL_PATH,
        vehicle_model_path=VEHICLE_MODEL_PATH,
    )

    processor.inference()
    processor.analyze()
    # Render only the advanced annotated output
    processor.render_video_advanced()
    # Path to the advanced output
    output_dir = Path("output") / video_name
    advanced = output_dir / "inference_advanced.mp4"
    # Generate violations CSV report
    csv_path: Path | None = None
    try:
        fps = float(processor.video_info.fps) if processor.video_info and processor.video_info.fps else None
        csv_path = output_dir / f"{video_name}_stop_detection_report.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["car_id", "status", "first_frame", "last_frame", "first_time_s", "last_time_s"])
            for car_id, data in (processor.analysis or {}).items():
                status = data.get("Status")
                if status and "Failed" in status:
                    first_f = processor.first_last_frames.get(car_id, {}).get("first")
                    last_f = processor.first_last_frames.get(car_id, {}).get("last")
                    first_t = (first_f / fps) if (fps and first_f is not None) else None
                    last_t = (last_f / fps) if (fps and last_f is not None) else None
                    writer.writerow([car_id, status, first_f, last_f, first_t, last_t])
    except Exception as e:
        logging.error(f"Failed to generate violations CSV: {e}")
        csv_path = None
    return advanced, csv_path


def is_valid_video(video_path: Path) -> bool:
    try:
        if not video_path.exists() or os.path.getsize(video_path) == 0:
            return False
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frames > 0
    except Exception:
        return False


def upload_results_to_gcs(bucket_name: str, output_video_path: str, csv_file_path: str | None, video_name: str) -> bool:
    """Upload processed results to Google Cloud Storage with better folder structure"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Upload processed video
        if output_video_path and os.path.exists(output_video_path):
            # Determine file extension based on actual file type
            if output_video_path.endswith('.avi'):
                video_blob = bucket.blob(f"processed-videos/{video_name}_processed.avi")
                logging.info(
                    f"Uploading processed video (AVI) to gs://{bucket_name}/processed-videos/{video_name}_processed.avi"
                )
            elif output_video_path.endswith('.mp4'):
                video_blob = bucket.blob(f"processed-videos/{video_name}_processed.mp4")
                logging.info(
                    f"Uploading processed video (MP4) to gs://{bucket_name}/processed-videos/{video_name}_processed.mp4"
                )
            else:
                # Fallback for other formats
                file_ext = os.path.splitext(output_video_path)[1]
                video_blob = bucket.blob(f"processed-videos/{video_name}_processed{file_ext}")
                logging.info(
                    f"Uploading processed video ({file_ext}) to gs://{bucket_name}/processed-videos/{video_name}_processed{file_ext}"
                )

            video_blob.upload_from_filename(output_video_path)

        # Upload CSV report (optional) under stop-detection-reports/
        if csv_file_path and os.path.exists(csv_file_path):
            csv_blob = bucket.blob(f"stop-detection-reports/{video_name}_stop_detection_report.csv")
            csv_blob.upload_from_filename(csv_file_path)
            logging.info(
                f"Uploaded CSV report to gs://{bucket_name}/stop-detection-reports/{video_name}_stop_detection_report.csv"
            )

        return True
    except Exception as e:
        logging.error(f"Error uploading results: {e}")
        return False


def upload_file_to_gcs(src_path: Path, bucket_name: str, dest_object: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_object)
    blob.content_type = "video/mp4"
    blob.upload_from_filename(str(src_path))
    logging.info(f"Uploaded {src_path} -> gs://{bucket_name}/{dest_object}")


def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    try:
        payload_raw = message.data.decode("utf-8") if message.data else "{}"
        payload = json.loads(payload_raw)

        bucket = payload.get("bucket")
        object_name = payload.get("name")

        if not bucket or not object_name:
            print("Skipping message without bucket/name fields")
            message.ack()
            return

        # Optional: ignore non-video objects defensively
        if not is_video_path(object_name):
            print(f"Ignoring non-video object: {object_name}")
            message.ack()
            return

        print(f"New finalized object detected: gs://{bucket}/{object_name}")

        # Compute destination object name upfront and skip if already processed (and non-empty)
        input_stem = Path(object_name).stem
        dest_filename = f"{input_stem}_processed.mp4"
        dest_obj = f"{DEST_PREFIX.rstrip('/')}/{dest_filename}"
        storage_client = storage.Client()
        dest_blob = storage_client.bucket(DEST_BUCKET).blob(dest_obj)
        if dest_blob.exists():
            try:
                dest_blob.reload()
                size = int(dest_blob.size or 0)
            except Exception:
                size = 0
            if size > 1024:  # treat >1KB as a valid processed artifact
                print(
                    f"Already processed (size={size} bytes). Skipping: gs://{DEST_BUCKET}/{dest_obj}"
                )
                message.ack()
                return
            else:
                print(
                    f"Existing processed object is too small (size={size}); will reprocess and overwrite."
                )

        local_path = DOWNLOAD_DIR / Path(object_name).name
        download_gcs_object(bucket, object_name, local_path)
        print(f"Downloaded to: {local_path}")

        # If AVI, convert to MP4 1280x720
        if local_path.suffix.lower() == ".avi":
            mp4_path = local_path.with_suffix(".mp4")
            ok = convert_avi_to_mp4(str(local_path), str(mp4_path))
            if not ok:
                raise RuntimeError("AVI to MP4 conversion failed; leaving message for retry")
            # Use the converted file for processing
            local_path = mp4_path

        if not os.path.exists(WHEEL_MODEL_PATH):
            raise FileNotFoundError(
                f"Wheel model not found at '{WHEEL_MODEL_PATH}'. Set WHEEL_MODEL_PATH env or place the file."
            )
        if not os.path.exists(VEHICLE_MODEL_PATH):
            raise FileNotFoundError(
                f"Vehicle model not found at '{VEHICLE_MODEL_PATH}'. Set VEHICLE_MODEL_PATH env or place the file."
            )

        advanced_out, csv_path = process_video(local_path)
        print(f"Processing completed for: {local_path}")

        # Validate output video before upload
        if not is_valid_video(advanced_out):
            raise RuntimeError(
                f"Advanced output invalid or empty: {advanced_out}. Not uploading."
            )

        # Upload using helper (names as <input_stem>_processed.<ext>)
        _ = upload_results_to_gcs(
            bucket_name=DEST_BUCKET,
            output_video_path=str(advanced_out),
            csv_file_path=str(csv_path) if csv_path else None,
            video_name=input_stem,
        )
        message.ack()
    except Exception as e:
        # Let Pub/Sub redeliver by not acking
        print(f"Error handling message: {e}")
        try:
            message.nack()
        except Exception:
            # If nack is unavailable for some reason, ensure we don't ack
            pass


def main() -> None:
    print(
        f"Starting subscriber for project='{PROJECT_ID}', subscription='{SUBSCRIPTION_ID}'.\n"
        "Ensure GOOGLE_APPLICATION_CREDENTIALS is set if running outside GCP."
    )

    # Ensure required model weights exist locally; download from GCS if missing
    try:
        if not os.path.exists(WHEEL_MODEL_PATH):
            logging.info(
                f"Wheel model not found locally at '{WHEEL_MODEL_PATH}'. Downloading from gs://{MODEL_BUCKET}/{WHEEL_MODEL_GCS_OBJECT}"
            )
            download_gcs_object(MODEL_BUCKET, WHEEL_MODEL_GCS_OBJECT, Path(WHEEL_MODEL_PATH))
        if not os.path.exists(VEHICLE_MODEL_PATH):
            logging.info(
                f"Vehicle model not found locally at '{VEHICLE_MODEL_PATH}'. Downloading from gs://{MODEL_BUCKET}/{VEHICLE_MODEL_GCS_OBJECT}"
            )
            download_gcs_object(MODEL_BUCKET, VEHICLE_MODEL_GCS_OBJECT, Path(VEHICLE_MODEL_PATH))
    except Exception as e:
        logging.error(f"Failed to ensure model weights locally: {e}")
        raise

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

    future = subscriber.subscribe(subscription_path, callback=callback)
    print(f"Listening for messages on {subscription_path}...")

    try:
        future.result()
    except KeyboardInterrupt:
        print("Stopping subscriber...")
        future.cancel()


if __name__ == "__main__":
    main()
