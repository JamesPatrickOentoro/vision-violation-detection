import os
import csv
import json
import shutil
import subprocess
import logging
from datetime import datetime
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
# Only handle notifications from this source bucket
SOURCE_BUCKET = os.environ.get("SOURCE_BUCKET", "vision-poc-bucket-adaro")
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

# Screenshot destination for violation frames
# Default bucket updated per request: adaro-vision-stop-detection/screenshoot
SCREENSHOT_BUCKET = os.environ.get("SCREENSHOT_BUCKET", "adaro-vision-stop-detection")
SCREENSHOT_PREFIX = os.environ.get("SCREENSHOT_PREFIX", "screenshoot/")

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
                "-an",                          # Drop audio to avoid decode errors
                "-movflags", "+faststart",     # Optional: faster playback start
                "-max_muxing_queue_size", "1024",  # Handle large queues
                output_path
            ]

            logging.info(f"Running FFmpeg command: {' '.join(cmd)}")

            # Run FFmpeg with timeout and capture output
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    timeout=900,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                logging.info("FFmpeg conversion completed (exit 0)")
            except subprocess.CalledProcessError as e:
                # Some damaged inputs can yield non-zero exit even if output is usable.
                # Accept the output if it validates as a playable video.
                err = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
                logging.warning("FFmpeg returned non-zero exit code; will validate output: %s", e.returncode)
                if err:
                    logging.debug("ffmpeg stderr (truncated): %s", err[-4000:])
                if is_valid_video(Path(output_path)):
                    logging.info("Output validated despite ffmpeg error; proceeding with converted file")
                else:
                    # Fallback attempt: drop audio and relax sync
                    fallback_cmd = [
                        "ffmpeg", "-y",
                        "-err_detect", "ignore_err",
                        "-fflags", "+genpts",
                        "-avoid_negative_ts", "make_zero",
                        "-i", input_path,
                        "-vf", "scale=1280:720",
                        "-b:v", "572k",
                        "-c:v", "libx264",
                        "-an",  # drop audio in fallback to avoid audio decode errors
                        "-vsync", "2",
                        output_path,
                    ]
                    logging.info("Retrying conversion without audio: %s", ' '.join(fallback_cmd))
                    subprocess.run(
                        fallback_cmd,
                        check=True,
                        timeout=900,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    logging.info("Fallback conversion completed")

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
    # Render only the advanced annotated output to a temporary AVI via cv2 writer
    tmp_out = Path("/tmp") / f"{video_name}_annot.avi"
    if tmp_out.exists():
        try:
            tmp_out.unlink()
        except Exception:
            pass
    processor.render_video_advanced_cv2(str(tmp_out))
    advanced = tmp_out
    # Validate; if invalid try to re-encode to MJPEG AVI via ffmpeg
    if not is_valid_video(advanced):
        logging.warning("Annotated AVI appears invalid; forcing re-encode to MJPEG AVI")
        advanced = reencode_avi(advanced)
    # Generate violations CSV report with: car_id, status, time(minute), screenshot(gs path)
    csv_path: Path | None = None
    try:
        fps = float(processor.video_info.fps) if processor.video_info and processor.video_info.fps else 30.0
        output_dir = Path("output") / video_name
        csv_path = output_dir / f"{video_name}_stop_detection_report.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare to read annotated video and capture screenshots
        cap = cv2.VideoCapture(str(advanced))
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["car_id", "status", "time", "screenshot"])  # simplified schema
            for car_id, data in (processor.analysis or {}).items():
                status = data.get("Status")
                if not status or "Failed" not in status:
                    continue
                # Frame when failure determined
                frame_idx = processor.car_left_zone.get(car_id)
                if frame_idx is None:
                    continue
                minute_mark = int((frame_idx / fps) // 60)

                # Extract frame from annotated video
                screenshot_path = None
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        local_ss = Path("/tmp") / f"{video_name}_car{car_id}_frame{frame_idx}.jpg"
                        try:
                            cv2.imwrite(str(local_ss), frame)
                            # Upload screenshot to GCS
                            ss_name = local_ss.name
                            ss_dest = f"{SCREENSHOT_PREFIX.rstrip('/')}/{ss_name}"
                            _upload_ok = upload_file_to_gcs_verified(local_ss, SCREENSHOT_BUCKET, ss_dest)
                            if _upload_ok:
                                screenshot_path = f"gs://{SCREENSHOT_BUCKET}/{ss_dest}"
                        except Exception as e:
                            logging.error(f"Failed to save/upload screenshot: {e}")

                writer.writerow([car_id, "Failed to stop", minute_mark, screenshot_path or ""])

        cap.release()
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


def faststart_mp4(input_path: Path) -> Path:
    """Remux MP4 to move moov atom to start for better streaming compatibility."""
    try:
        if shutil.which('ffmpeg') is None:
            return input_path  # Can't fix without ffmpeg
        fixed = input_path.with_name(input_path.stem + "_fixed.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-c", "copy",
            "-movflags", "+faststart",
            str(fixed),
        ]
        subprocess.run(cmd, check=True, timeout=300, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if is_valid_video(fixed):
            return fixed
        return input_path
    except Exception:
        return input_path


def reencode_mp4(input_path: Path, target_fps: int = 30) -> Path:
    """Force re-encode MP4 with stable timebase and pixel format to ensure playability."""
    try:
        if shutil.which('ffmpeg') is None:
            return input_path
        out = input_path.with_name(input_path.stem + "_reenc.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"fps={target_fps}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(out),
        ]
        subprocess.run(cmd, check=True, timeout=900, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if is_valid_video(out):
            return out
        return input_path
    except Exception:
        return input_path


def reencode_avi(input_path: Path, target_fps: int = 30) -> Path:
    """Re-encode to AVI (MJPEG) to ensure valid container when writer output is problematic."""
    try:
        if shutil.which('ffmpeg') is None:
            return input_path
        out = input_path.with_name(input_path.stem + "_reenc.avi")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"fps={target_fps}",
            "-c:v", "mjpeg",
            "-qscale:v", "5",
            "-an",
            str(out),
        ]
        subprocess.run(cmd, check=True, timeout=900, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if is_valid_video(out):
            return out
        return input_path
    except Exception:
        return input_path


def create_storage_client() -> storage.Client:
    return storage.Client()


def upload_results_to_gcs(bucket_name: str, output_video_path: str, csv_file_path: str | None, video_name: str, max_retries: int = 3) -> bool:
    """Upload processed results to Google Cloud Storage with retries and verification"""

    def verify_upload(blob: storage.Blob, local_file: str) -> bool:
        if not blob.exists():
            return False
        try:
            blob.reload()
        except Exception:
            pass
        local_size = os.path.getsize(local_file)
        remote_size = int(getattr(blob, 'size', 0) or 0)
        return remote_size == local_size and remote_size > 0

    def upload_with_retry(blob: storage.Blob, filepath: str, file_type: str = "file") -> bool:
        # Set a chunk size to force resumable uploads for larger files
        blob.chunk_size = 8 * 1024 * 1024  # 8 MB

        for attempt in range(max_retries):
            try:
                # Content type mapping
                content_type = (
                    'video/mp4' if filepath.endswith('.mp4') else
                    'video/x-msvideo' if filepath.endswith('.avi') else
                    'image/jpeg' if filepath.endswith('.jpg') or filepath.endswith('.jpeg') else
                    'image/png' if filepath.endswith('.png') else
                    'text/csv' if filepath.endswith('.csv') else
                    'application/octet-stream'
                )
                blob.content_type = content_type
                blob.metadata = {
                    'uploaded_by': 'stop_detection_system',
                    'upload_timestamp': datetime.now().isoformat(),
                    'original_filename': os.path.basename(filepath),
                }

                blob.upload_from_filename(filepath)

                if verify_upload(blob, filepath):
                    logging.info(
                        f"Successfully uploaded and verified {file_type} to gs://{bucket_name}/{blob.name}"
                    )
                    return True
                else:
                    logging.warning(
                        f"Upload verification failed for {file_type}, attempt {attempt + 1}/{max_retries}"
                    )
            except Exception as e:
                logging.warning(
                    f"Upload attempt {attempt + 1} failed for {file_type}: {e}"
                )
                if attempt == max_retries - 1:
                    raise
        return False

    try:
        storage_client = create_storage_client()
        bucket = storage_client.bucket(bucket_name)

        # Upload processed video
        if output_video_path and os.path.exists(output_video_path):
            file_ext = os.path.splitext(output_video_path)[1]
            video_blob = bucket.blob(f"processed-videos/{video_name}_processed{file_ext}")
            if not upload_with_retry(video_blob, output_video_path, "video"):
                raise Exception("Video upload failed after retries")

        # Upload CSV report (optional) under stop-detection-reports/
        if csv_file_path and os.path.exists(csv_file_path):
            csv_blob = bucket.blob(f"stop-detection-reports/{video_name}_stop_detection_report.csv")
            if not upload_with_retry(csv_blob, csv_file_path, "CSV report"):
                raise Exception("CSV report upload failed after retries")

        return True
    except Exception as e:
        msg = str(e)
        if "Provided scope(s) are not authorized" in msg or "insufficientPermissions" in msg:
            logging.error(
                "GCS upload failed due to missing OAuth scopes. Ensure one of: "
                "1) Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON with Storage permissions; "
                "2) If running on GCE/GKE, grant the VM/pod service account Storage roles and add the 'Cloud Platform' scope; "
                "3) Or run `gcloud auth application-default login` for local ADC."
            )
        logging.error(f"Error uploading results: {e}")
        return False


def upload_file_to_gcs_verified(local_path: Path, bucket_name: str, dest_object: str, max_retries: int = 3) -> bool:
    """Upload a single file with retries + size verification. Returns True on success."""
    storage_client = create_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(dest_object)

    # Inner helpers duplicated to keep function independent
    def verify_upload(b: storage.Blob, path: str) -> bool:
        if not b.exists():
            return False
        try:
            b.reload()
        except Exception:
            pass
        local_size = os.path.getsize(path)
        remote_size = int(getattr(b, 'size', 0) or 0)
        return remote_size == local_size and remote_size > 0

    blob.chunk_size = 8 * 1024 * 1024
    for attempt in range(max_retries):
        try:
            # Content type
            name = str(local_path)
            content_type = (
                'video/mp4' if name.endswith('.mp4') else
                'video/x-msvideo' if name.endswith('.avi') else
                'image/jpeg' if name.endswith('.jpg') or name.endswith('.jpeg') else
                'image/png' if name.endswith('.png') else
                'text/csv' if name.endswith('.csv') else
                'application/octet-stream'
            )
            blob.content_type = content_type
            blob.metadata = {
                'uploaded_by': 'stop_detection_system',
                'upload_timestamp': datetime.now().isoformat(),
                'original_filename': os.path.basename(name),
            }
            blob.upload_from_filename(str(local_path))
            if verify_upload(blob, str(local_path)):
                return True
            logging.warning(f"Verification failed for {dest_object}, attempt {attempt+1}/{max_retries}")
        except Exception as e:
            logging.warning(f"Upload attempt {attempt+1} failed for {dest_object}: {e}")
            if attempt == max_retries - 1:
                return False
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

        # Gate by expected source bucket to avoid processing unexpected notifications
        if bucket != SOURCE_BUCKET:
            print(f"Skipping message from non-source bucket '{bucket}'. Expected '{SOURCE_BUCKET}'.")
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

        # Validate output video before upload (re-encode was attempted in process_video if needed)
        if not is_valid_video(advanced_out):
            # As a last resort, try a final re-encode here
            logging.warning("Advanced output still invalid before upload; final re-encode attempt")
            advanced_out = reencode_mp4(advanced_out)
            if not is_valid_video(advanced_out):
                raise RuntimeError(
                    f"Advanced output invalid or empty after re-encode: {advanced_out}. Not uploading."
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
