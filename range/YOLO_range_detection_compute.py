import os
import cv2
import numpy as np
from ultralytics import YOLO
import csv
from datetime import datetime
import shutil
from google.cloud import storage, pubsub_v1
import base64
import json
import traceback
import time
from concurrent.futures import TimeoutError
import logging
import threading
import torch
import subprocess
import tempfile
from urllib.parse import unquote

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yolo-worker.log'),
        logging.StreamHandler()
    ]
)

# --- Configuration ---
PROJECT_ID = "adaro-vision-poc"
SOURCE_BUCKET_NAME = "vision-poc-bucket-adaro"
DESTINATION_BUCKET_NAME = "adaro-vision-range-detection"
SUBSCRIPTION_ID = "adaro-range-detection-sub"

YOLO_MODEL_FILENAME = "yolo11s.pt"
HOMOGRAPHY_FILENAME = "homography.npy"
METER_PER_PIXEL = 0.1096
DISTANCE_THRESHOLD_METERS = 10.0
YOLO_CONFIDENCE_THRESHOLD = 0.4


def format_timestamp_ms(timestamp_ms: float) -> str:
    """Convert a millisecond timestamp to H:MM:SS for human-readable output."""
    if timestamp_ms is None:
        return "0:00:00"

    total_milliseconds = max(0, int(round(timestamp_ms)))
    hours = total_milliseconds // 3_600_000
    minutes = (total_milliseconds % 3_600_000) // 60_000
    seconds = (total_milliseconds % 60_000) // 1_000
    return f"{hours}:{minutes:02d}:{seconds:02d}"

# --- Global Variables for Models ---
model = None
H = None
device = None

# --- Global Variables for Duplicate Detection ---
processed_videos = set()  # Track which videos we've already processed
processing_videos = set()  # Track which videos are currently being processed
corrupted_videos = set()  # Track which videos are too corrupted to process
MAX_PROCESSED_VIDEOS = 1000  # Limit memory usage
processing_lock = threading.Lock()  # Thread lock for safe access
global_processing_lock = threading.Lock()  # Global lock to ensure only one video processes at a time

# --- GPU Configuration Functions ---
def setup_gpu():
    """Configure GPU settings for optimal performance"""
    global device
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        
        logging.info(f"🚀 GPU detected: {gpu_name}")
        logging.info(f"📊 Available GPUs: {gpu_count}")
        
        # Set CUDA device
        torch.cuda.set_device(0)
        
        # Optimize CUDA performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory fraction to avoid OOM errors
        torch.cuda.empty_cache()
        
        logging.info(f"✅ GPU configured successfully on device: {device}")
        return True
    else:
        device = torch.device('cpu')
        logging.warning("⚠️  No GPU detected, using CPU. Performance will be slower.")
        return False

def get_device_info():
    """Get detailed device information"""
    if device and device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3  # GB
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3  # GB
        
        logging.info(f"🎯 GPU Memory: {gpu_memory:.1f}GB total, {allocated_memory:.1f}GB allocated, {cached_memory:.1f}GB cached")
        return f"GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)"
    else:
        return "CPU"

# --- Video Conversion Functions ---
def convert_avi_to_mp4(input_path, output_path):
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
            subprocess.run(cmd, check=True, timeout=300)
            logging.info("✅ FFmpeg conversion successful")
            
            # Verify the converted file has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logging.info(f"✅ Converted MP4 file size: {file_size} bytes")
                
                # Additional validation: check if video can be opened
                test_cap = cv2.VideoCapture(output_path)
                if test_cap.isOpened():
                    frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    test_cap.release()
                    if frame_count > 0:
                        logging.info(f"✅ Converted MP4 verified: {frame_count} frames")
                        return True
                    else:
                        logging.error("❌ Converted MP4 has 0 frames")
                        return False
                else:
                    logging.error("❌ Converted MP4 cannot be opened")
                    return False
            else:
                logging.error("❌ Converted MP4 file is empty or missing")
                return False
        else:
            logging.error("❌ FFmpeg not found! Please install FFmpeg: sudo apt-get install ffmpeg")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("❌ FFmpeg conversion timed out after 5 minutes")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ FFmpeg failed with return code {e.returncode}")
        logging.error(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"❌ Error during FFmpeg conversion: {e}")
        return False


def extract_frames_with_ffmpeg(input_path: str, frames_dir: str, target_fps: int = 25) -> int:
    """Extract frames using FFmpeg into frames_dir. Returns number of frames extracted."""
    try:
        os.makedirs(frames_dir, exist_ok=True)
        if shutil.which('ffmpeg') is None:
            logging.error("❌ FFmpeg not found for frame extraction")
            return 0
        
        # Enhanced FFmpeg command for corrupted videos
        cmd = [
            'ffmpeg', '-y',
            '-fflags', '+genpts',
            '-avoid_negative_ts', 'make_zero',
            '-err_detect', 'ignore_err',
            '-i', input_path,
            '-vsync', '2',
            '-r', str(target_fps),
            '-f', 'image2',
            '-q:v', '2',  # High quality JPEG
            '-skip_frame', 'nokey',  # Skip corrupted frames
            '-threads', '0',  # Use all available threads
            os.path.join(frames_dir, 'frame_%06d.jpg')
        ]
        logging.info(f"Extracting frames with FFmpeg (corruption-resistant): {' '.join(cmd)}")
        subprocess.run(cmd, check=True, timeout=600)
        
        # Count frames and validate them
        frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')]
        valid_frames = 0
        for frame_file in frame_files:
            frame_path = os.path.join(frames_dir, frame_file)
            try:
                # Check if frame is valid by trying to read it
                test_img = cv2.imread(frame_path)
                if test_img is not None and test_img.shape[0] > 0 and test_img.shape[1] > 0:
                    valid_frames += 1
                else:
                    # Remove invalid frame
                    os.remove(frame_path)
            except Exception:
                # Remove corrupted frame
                try:
                    os.remove(frame_path)
                except:
                    pass
        
        logging.info(f"FFmpeg extracted {valid_frames} valid frames (removed {len(frame_files) - valid_frames} corrupted)")
        return valid_frames
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ FFmpeg frame extraction failed: {e.stderr}")
        return 0
    except Exception as e:
        logging.error(f"❌ Error during frame extraction: {e}")
        return 0

def convert_avi_to_mp4_opencv(input_path, output_path):
    """Fallback conversion using OpenCV (less reliable but available)"""
    try:
        logging.info("Converting AVI to MP4 using OpenCV...")
        
        # Open the AVI file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.error("Could not open AVI file for conversion")
            return False
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_value = cap.get(cv2.CAP_PROP_FPS)
        try:
            fps = int(fps_value) if fps_value and fps_value > 0 else 25
        except Exception:
            fps = 25
        if fps == 25:
            logging.warning("FPS reported as 0/invalid; falling back to 25 FPS for writer")

        fps_float = float(fps if fps > 0 else 25)
        
        # Create MP4 writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logging.info(f"Converted {frame_count} frames...")
        
        cap.release()
        out.release()
        
        logging.info(f"OpenCV conversion completed: {frame_count} frames")
        return True
        
    except Exception as e:
        logging.error(f"Error during OpenCV conversion: {e}")
        return False

def convert_mp4_to_avi(input_path, output_path):
    """Convert MP4 video to AVI format using FFmpeg"""
    try:
        # Check if FFmpeg is available
        if shutil.which('ffmpeg') is not None:
            logging.info("Converting MP4 to AVI using FFmpeg...")
            
            cmd = [
                'ffmpeg',
                '-y',               # Overwrite output file without asking
                '-i', input_path,
                '-c:v', 'libx264',  # Use H.264 codec for better compatibility
                '-c:a', 'aac',      # Use AAC audio codec
                '-f', 'avi',        # Force AVI format
                '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
                '-fflags', '+genpts',              # Generate presentation timestamps
                '-err_detect', 'ignore_err',       # Ignore errors and continue
                output_path
            ]
            
            logging.info(f"Running FFmpeg command: {' '.join(cmd)}")
            
            # Run FFmpeg with timeout and capture output
            subprocess.run(cmd, check=True, timeout=300)
            logging.info("✅ FFmpeg MP4 to AVI conversion successful")
            
            # Verify the converted file has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size = os.path.getsize(output_path)
                logging.info(f"✅ Converted AVI file size: {file_size} bytes")
                
                # Additional validation: check if video can be opened
                test_cap = cv2.VideoCapture(output_path)
                if test_cap.isOpened():
                    frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    test_cap.release()
                    if frame_count > 0:
                        logging.info(f"✅ Converted AVI verified: {frame_count} frames")
                        return True
                    else:
                        logging.error("❌ Converted AVI has 0 frames")
                        return False
                else:
                    logging.error("❌ Converted AVI cannot be opened")
                    return False
            else:
                logging.error("❌ Converted AVI file is empty or missing")
                return False
        else:
            logging.error("❌ FFmpeg not found! Please install FFmpeg: sudo apt-get install ffmpeg")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("❌ FFmpeg conversion timed out after 5 minutes")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ FFmpeg MP4 to AVI conversion failed with return code {e.returncode}")
        logging.error(f"FFmpeg stderr: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"❌ Error during FFmpeg MP4 to AVI conversion: {e}")
        return False

def convert_mp4_to_avi_opencv(input_path, output_path):
    """Fallback MP4 to AVI conversion using OpenCV"""
    try:
        logging.info("Converting MP4 to AVI using OpenCV...")
        
        # Open the MP4 file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.error("Could not open MP4 file for conversion")
            return False
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_value = cap.get(cv2.CAP_PROP_FPS)
        try:
            fps = int(fps_value) if fps_value and fps_value > 0 else 25
        except Exception:
            fps = 25
        if fps == 25:
            logging.warning("FPS reported as 0/invalid; falling back to 25 FPS for writer")
        
        # Create AVI writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logging.info(f"Converted {frame_count} frames...")
        
        cap.release()
        out.release()
        
        logging.info(f"OpenCV MP4 to AVI conversion completed: {frame_count} frames")
        return True
        
    except Exception as e:
        logging.error(f"Error during OpenCV MP4 to AVI conversion: {e}")
        return False

def ensure_mp4_format(video_path, video_name):
    """Ensure video is in MP4 format, convert if necessary (improved reliability)"""
    # Check if video is already MP4
    if video_path.lower().endswith('.mp4'):
        logging.info(f"✅ Video {video_name} is already in MP4 format")
        return video_path, False  # False means no conversion needed
    
    # Check if it's AVI and needs conversion
    if video_path.lower().endswith('.avi'):
        logging.info(f"🔄 Converting AVI video {video_name} to MP4 format for better compatibility...")
        
        # Create temporary MP4 file with unique name to avoid conflicts
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False, prefix=f'conversion_{int(time.time())}_') as tmp_mp4:
            mp4_path = tmp_mp4.name
        
        logging.info(f"📁 Temporary MP4 path: {mp4_path}")
        
        # Convert AVI to MP4
        if convert_avi_to_mp4(video_path, mp4_path):
            logging.info(f"✅ Successfully converted {video_name} to MP4")
            
            # Verify the converted file can be opened
            test_cap = cv2.VideoCapture(mp4_path)
            if test_cap.isOpened():
                test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                test_cap.release()
                logging.info(f"✅ Converted MP4 verified: {test_frames} frames readable")
                return mp4_path, True  # True means conversion was done
            else:
                logging.error(f"❌ Converted MP4 cannot be opened by OpenCV")
                return video_path, False
        else:
            logging.error(f"❌ Failed to convert {video_name} to MP4")
            logging.warning(f"⚠️ AVI file appears corrupted - checking if it's too corrupted to process")
            
            # Check if video is too corrupted to process
            try:
                test_cap = cv2.VideoCapture(video_path)
                if test_cap.isOpened():
                    # Count readable frames in first 200 frames
                    readable_frames = 0
                    corrupted_frames = 0
                    for i in range(200):  # Test first 200 frames
                        ret, frame = test_cap.read()
                        if ret and frame is not None and np.mean(frame) > 5:
                            readable_frames += 1
                        else:
                            corrupted_frames += 1
                    test_cap.release()
                    
                    corruption_rate = (corrupted_frames / 200) * 100
                    logging.info(f"🔍 Corruption check: {corruption_rate:.1f}% of first 200 frames are corrupted")
                    
                    if corruption_rate > 95.0 or readable_frames < 5:
                        logging.error(f"❌ Video too heavily corrupted: {corruption_rate:.1f}% corruption, only {readable_frames} readable frames")
                        logging.error(f"🚫 Skipping this video due to severe corruption")
                        return None, False  # Signal to skip this video
                    elif readable_frames > 10:
                        logging.info(f"⚠️ AVI has {readable_frames} readable frames - processing with error recovery")
                        return video_path, False  # Process original with recovery
                    else:
                        logging.error(f"❌ AVI file too corrupted - only {readable_frames} frames readable")
                        return None, False
                else:
                    logging.error(f"❌ Cannot open corrupted AVI file")
                    return None, False
            except Exception as e:
                logging.error(f"❌ Error testing corrupted AVI: {e}")
                return None, False
    
    # For other formats, try to process as-is
    logging.info(f"⚠️ Video {video_name} is in unsupported format, attempting to process directly")
    return video_path, False

# --- Cloud Storage Helper Functions ---
def download_from_gcs(source_blob_name, destination_file_path):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(SOURCE_BUCKET_NAME)
        blob = bucket.blob(source_blob_name)
        os.makedirs(os.path.dirname(destination_file_path), exist_ok=True)
        # Early existence check for clearer error messages
        if not blob.exists():
            logging.error(f"GCS object not found: bucket={SOURCE_BUCKET_NAME}, name={repr(source_blob_name)}")
            raise FileNotFoundError(f"GCS object does not exist: gs://{SOURCE_BUCKET_NAME}/{source_blob_name}")
        blob.download_to_filename(destination_file_path)
        logging.info(f"Successfully downloaded {source_blob_name} to {destination_file_path}")
    except Exception as e:
        logging.error(f"Error downloading {source_blob_name} from GCS: {e}")
        raise

def upload_to_gcs(source_file_path, destination_blob_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(DESTINATION_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        logging.info(f"Successfully uploaded {source_file_path} to gs://{DESTINATION_BUCKET_NAME}/{destination_blob_name}")
    except Exception as e:
        logging.error(f"Error uploading {source_file_path} to GCS: {e}")
        raise

def upload_results_to_gcs(bucket_name, output_video_path, csv_file_path, video_name):
    """Upload processed results to Google Cloud Storage with better folder structure"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # Upload processed video
        if os.path.exists(output_video_path):
            # Determine file extension based on actual file type
            if output_video_path.endswith('.avi'):
                video_blob = bucket.blob(f"processed-videos/{video_name}_processed.avi")
                logging.info(f"Uploaded processed video (AVI) to gs://{bucket_name}/processed-videos/{video_name}_processed.avi")
            elif output_video_path.endswith('.mp4'):
                video_blob = bucket.blob(f"processed-videos/{video_name}_processed.mp4")
                logging.info(f"Uploaded processed video (MP4) to gs://{bucket_name}/processed-videos/{video_name}_processed.mp4")
            else:
                # Fallback for other formats
                file_ext = os.path.splitext(output_video_path)[1]
                video_blob = bucket.blob(f"processed-videos/{video_name}_processed{file_ext}")
                logging.info(f"Uploaded processed video ({file_ext}) to gs://{bucket_name}/processed-videos/{video_name}_processed{file_ext}")
            
            video_blob.upload_from_filename(output_video_path)
        
        # Upload CSV report
        if os.path.exists(csv_file_path):
            csv_blob = bucket.blob(f"range-detection-reports/{video_name}_range_detection_report.csv")
            csv_blob.upload_from_filename(csv_file_path)
            logging.info(f"Uploaded CSV report to gs://{bucket_name}/range-detection-reports/{video_name}_range_detection_report.csv")
        
        return True
    except Exception as e:
        logging.error(f"Error uploading results: {e}")
        return False

# --- Helper and Processing Functions ---
def scale_annotations_for_resolution(frame_width, frame_height, target_width=1280, target_height=720):
    """Calculate scaling factors to fit annotations for target resolution"""
    scale_x = target_width / frame_width
    scale_y = target_height / frame_height
    
    # Use uniform scaling to maintain aspect ratio
    scale = min(scale_x, scale_y)
    
    # Calculate offset to center the scaled content
    new_width = int(frame_width * scale)
    new_height = int(frame_height * scale)
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    
    logging.info(f"📐 Scaling annotations: {frame_width}x{frame_height} → {target_width}x{target_height}")
    logging.info(f"📐 Scale factor: {scale:.3f}, Offset: ({offset_x}, {offset_y})")
    
    return scale, offset_x, offset_y

def scale_bbox(bbox, scale, offset_x, offset_y):
    """Scale bounding box coordinates for target resolution"""
    x, y, w, h = bbox
    scaled_x = int(x * scale + offset_x)
    scaled_y = int(y * scale + offset_y)
    scaled_w = int(w * scale)
    scaled_h = int(h * scale)
    return (scaled_x, scaled_y, scaled_w, scaled_h)

def compute_distance_with_homography(bbox1, bbox2, homography_matrix, meter_per_pixel=1.0):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    pt1 = np.array([[[x1 + w1 / 2, y1 + h1 / 2]]], dtype='float32')
    pt2 = np.array([[[x2 + w2 / 2, y2 + h2 / 2]]], dtype='float32')
    topdown_pt1 = cv2.perspectiveTransform(pt1, homography_matrix)[0][0]
    topdown_pt2 = cv2.perspectiveTransform(pt2, homography_matrix)[0][0]
    pixel_distance = np.linalg.norm(topdown_pt1 - topdown_pt2)
    distance_m = pixel_distance * meter_per_pixel
    return distance_m

def process_video(video_blob_name, model_instance, h_matrix):
    start_time = time.time()
    logging.info(f"🚀 Starting processing for {video_blob_name}")
    temp_dir = f"/tmp/{video_blob_name.replace('/', '_')}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize last good frame for interpolation
    process_video.last_good_frame = None
    
    video_basename = os.path.basename(video_blob_name)
    local_input_video_path = os.path.join(temp_dir, video_basename)
    video_name_without_ext = os.path.splitext(video_basename)[0]
    csv_report_temp_path = os.path.join(temp_dir, f"report_{video_name_without_ext}.csv")
    report_date = datetime.now().strftime('%Y-%m-%d')
    event_counter = 1

    def log_report_entry(csv_writer_obj, timestamp_ms: float, did_violate: str, vehicle_identifier=None):
        nonlocal event_counter

        timestamp_label = format_timestamp_ms(timestamp_ms)
        vehicle_id_value = vehicle_identifier if vehicle_identifier is not None else event_counter

        csv_writer_obj.writerow([
            report_date,
            video_basename,
            timestamp_label,
            vehicle_id_value,
            did_violate
        ])

        if vehicle_identifier is None:
            event_counter += 1

    # Variables to track conversion status
    input_was_converted = False
    temp_mp4_path = None
    final_output_path = None

    try:
        # Download the video from GCS
        download_start = time.time()
        download_from_gcs(video_blob_name, local_input_video_path)
        download_time = time.time() - download_start
        logging.info(f"📥 Download completed in {download_time:.2f} seconds")
        
        # Ensure video is in MP4 format for processing
        conversion_start = time.time()
        processed_input_path, input_was_converted = ensure_mp4_format(local_input_video_path, video_name_without_ext)
        conversion_time = time.time() - conversion_start
        if input_was_converted:
            logging.info(f"🔄 Video conversion completed in {conversion_time:.2f} seconds")
        
        if processed_input_path is None:
            logging.error(f"❌ Video {video_blob_name} is too heavily corrupted to process")
            logging.error(f"🚫 Creating skip report and acknowledging message")
            
            # Create a skip report CSV
            with open(csv_report_temp_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['Date', 'Video Name', 'Timestamp On Video (ms)', 'Vehicle ID', 'Did Violate'])
                log_report_entry(csv_writer, 0, 'N/A', vehicle_identifier='N/A')
            
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
            upload_results_to_gcs(DESTINATION_BUCKET_NAME, skip_video_path, csv_report_temp_path, video_name_without_ext)
            
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
            
            return  # Exit early, acknowledge the message
        
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
        
        # Create temporary MP4 output for processing at target resolution
        temp_mp4_output = os.path.join(temp_dir, f"processed_{video_name_without_ext}_temp.mp4")
        
        # Create output video writer - Use MP4 format for processing at target resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4V codec for MP4
        out = cv2.VideoWriter(temp_mp4_output, fourcc, fps, (target_width, target_height))
        
        if not out.isOpened():
            logging.warning("MP4V codec failed, trying H264 codec...")
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # H264 codec for MP4
            out = cv2.VideoWriter(temp_mp4_output, fourcc, fps, (target_width, target_height))
            
        if not out.isOpened():
            logging.warning("H264 codec failed, falling back to XVID AVI...")
            # Fallback to AVI if MP4 fails
            temp_mp4_output = os.path.join(temp_dir, f"processed_{video_name_without_ext}_temp.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_mp4_output, fourcc, fps, (target_width, target_height))

        logging.info(f"🎬 Processing video frames for {video_blob_name}...")
        frame_processing_start = time.time()
        with open(csv_report_temp_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Date', 'Video Name', 'Timestamp On Video (ms)', 'Vehicle ID', 'Did Violate'])

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

            # Corrupted frame handling variables
            max_retry_attempts = 3
            consecutive_corrupted = 0
            max_consecutive_corrupted = 10  # Skip ahead after 10 consecutive corrupted frames
            
            while current_frame_index < total_frames:
                frame_retrieved = False
                retry_count = 0
                
                # Try to get the frame with retries
                while retry_count < max_retry_attempts and not frame_retrieved:
                    grabbed = cap.grab()
                    if grabbed:
                        ret, frame = cap.retrieve()
                        if ret and frame is not None:
                            # Validate frame - check if it's not completely black or corrupted
                            if np.mean(frame) > 5 and frame.shape[0] > 0 and frame.shape[1] > 0:
                                frame_retrieved = True
                                total_good_frames += 1
                                consecutive_corrupted = 0
                            else:
                                retry_count += 1
                                if retry_count < max_retry_attempts:
                                    logging.warning(f"Frame {current_frame_index} appears corrupted (black/invalid), retry {retry_count}/{max_retry_attempts}")
                        else:
                            retry_count += 1
                            if retry_count < max_retry_attempts:
                                logging.warning(f"Frame {current_frame_index} could not be retrieved, retry {retry_count}/{max_retry_attempts}")
                    else:
                        retry_count += 1
                        if retry_count < max_retry_attempts:
                            logging.warning(f"Frame {current_frame_index} could not be grabbed, retry {retry_count}/{max_retry_attempts}")
                
                # If all retries failed, use frame interpolation or black frame
                if not frame_retrieved:
                    total_bad_frames += 1
                    consecutive_corrupted += 1
                    
                    # Try to use the last good frame instead of black frame for better continuity
                    if hasattr(process_video, 'last_good_frame') and process_video.last_good_frame is not None:
                        frame = process_video.last_good_frame.copy()
                        logging.warning(f"Frame {current_frame_index} corrupted, using previous good frame")
                    else:
                        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                        logging.warning(f"Frame {current_frame_index} corrupted after {max_retry_attempts} retries, using black frame")
                    
                    if consecutive_corrupted >= max_consecutive_corrupted:
                        # Skip ahead to find next good frame
                        skip_frames = min(50, max_consecutive_corrupted * 2)  # Skip more frames
                        new_position = min(current_frame_index + skip_frames, total_frames - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_position)
                        current_frame_index = new_position
                        consecutive_corrupted = 0
                        logging.warning(f"⏭️  Skipped {skip_frames} frames due to corruption, jumping to frame {new_position}")
                else:
                    # Store the last good frame for interpolation
                    process_video.last_good_frame = frame.copy()

                frame_timestamp_ms = (current_frame_index / fps_float) * 1000.0
                
                # Log progress every 100 frames
                if current_frame_index % 100 == 0:
                    logging.info(
                        f"Processed {current_frame_index}/{total_frames} frames (good: {total_good_frames}, "
                        f"bad: {total_bad_frames}), current timestamp: {frame_timestamp_ms / 1000.0:.2f}s"
                    )
                
                # Use GPU for inference if available
                if device and device.type == 'cuda':
                    results = model_instance(frame, verbose=False, device=device, conf=YOLO_CONFIDENCE_THRESHOLD)[0]
                else:
                    results = model_instance(frame, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)[0]
                
                detections = []
                for det in results.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) in [2, 5, 7]:  # car, bus, truck
                        w, h = x2 - x1, y2 - y1
                        detections.append(((x1, y1, w, h), int(cls)))

                # Create output frame at target resolution
                output_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                
                # Scale and center the original frame
                scaled_frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))
                output_frame[offset_y:offset_y + scaled_frame.shape[0], offset_x:offset_x + scaled_frame.shape[1]] = scaled_frame

                # Draw all detections first (bounding boxes and labels)
                for (bbox, cls) in detections:
                    scaled_bbox = scale_bbox(bbox, scale, offset_x, offset_y)
                    x, y, w, h = scaled_bbox
                    label = model_instance.names[cls]
                    cv2.rectangle(output_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                    cv2.putText(output_frame, label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Then draw distance lines and measurements for pairs under threshold
                if len(detections) >= 2:
                    for i in range(len(detections)):
                        for j in range(i + 1, len(detections)):
                            (bbox1, cls1), (bbox2, cls2) = detections[i], detections[j]
                            distance = compute_distance_with_homography(bbox1, bbox2, h_matrix, METER_PER_PIXEL)
                            did_violate = "Yes" if distance < DISTANCE_THRESHOLD_METERS else "No"

                            # Log violation status for each detection pair
                            log_report_entry(csv_writer, frame_timestamp_ms, did_violate)

                            # Draw distance line and measurement only if under threshold
                            if did_violate == "Yes":
                                # Scale bounding box coordinates for target resolution
                                scaled_bbox1 = scale_bbox(bbox1, scale, offset_x, offset_y)
                                scaled_bbox2 = scale_bbox(bbox2, scale, offset_x, offset_y)
                                
                                cx1, cy1 = int(scaled_bbox1[0] + scaled_bbox1[2] / 2), int(scaled_bbox1[1] + scaled_bbox1[3] / 2)
                                cx2, cy2 = int(scaled_bbox2[0] + scaled_bbox2[2] / 2), int(scaled_bbox2[1] + scaled_bbox2[3] / 2)
                                
                                cv2.line(output_frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
                                mid_x, mid_y = int((cx1 + cx2) / 2), int((cy1 + cy2) / 2)
                                cv2.putText(output_frame, f"{distance:.2f} m", (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                out.write(output_frame)
                current_frame_index += 1
        
        frame_processing_time = time.time() - frame_processing_start
        logging.info(f"🎬 Frame processing completed in {frame_processing_time:.2f} seconds")
        logging.info(f"📊 Frame processing summary: good={total_good_frames}, bad={total_bad_frames}, fps_used={fps}")
        
        # Calculate corruption statistics
        corruption_rate = (total_bad_frames / (total_good_frames + total_bad_frames)) * 100 if (total_good_frames + total_bad_frames) > 0 else 0
        logging.info(f"🔧 Corruption handling: {corruption_rate:.1f}% frames were corrupted/skipped")
        
        # Process whatever good frames we have, regardless of corruption level
        if corruption_rate > 50.0:
            logging.warning(f"⚠️  High corruption detected: {corruption_rate:.1f}% corrupted frames")
            logging.warning(f"📊 Processing {total_good_frames} good frames out of {total_good_frames + total_bad_frames} total frames")
        else:
            logging.info(f"✅ Low corruption level: {corruption_rate:.1f}% corrupted frames")
        
        if total_good_frames > 0:
            frame_fps = total_good_frames / frame_processing_time
            logging.info(f"⚡ Frame processing speed: {frame_fps:.2f} frames/second")

        # If too few good frames processed, try FFmpeg-based extraction fallback
        if total_good_frames < 10 and total_frames > 10:
            logging.warning("Too few frames processed via OpenCV; attempting FFmpeg frame extraction fallback")
            frames_dir = os.path.join(temp_dir, 'extracted_frames')
            extracted = extract_frames_with_ffmpeg(processed_input_path, frames_dir, target_fps=fps if fps > 0 else 25)
            if extracted > 0:
                logging.info(f"FFmpeg extracted {extracted} frames, processing with full annotation logic...")
                # Reopen CSV in append mode
                with open(csv_report_temp_path, 'a', newline='') as csv_file2:
                    csv_writer2 = csv.writer(csv_file2)
                    for idx in range(1, extracted + 1):
                        frame_path = os.path.join(frames_dir, f'frame_{idx:06d}.jpg')
                        if not os.path.exists(frame_path):
                            continue
                        frame = cv2.imread(frame_path)
                        if frame is None:
                            continue
                        
                        # Calculate scaling factors for this frame
                        frame_height, frame_width = frame.shape[:2]
                        scale, offset_x, offset_y = scale_annotations_for_resolution(frame_width, frame_height, target_width, target_height)
                        
                        # Create output frame at target resolution
                        output_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                        
                        # Scale and center the original frame
                        scaled_frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))
                        output_frame[offset_y:offset_y + scaled_frame.shape[0], offset_x:offset_x + scaled_frame.shape[1]] = scaled_frame
                        
                        # Run detection
                        if device and device.type == 'cuda':
                            results = model_instance(frame, verbose=False, device=device, conf=YOLO_CONFIDENCE_THRESHOLD)[0]
                        else:
                            results = model_instance(frame, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)[0]
                        
                        # Process detections with full annotation logic
                        frame_timestamp_ms = ((idx - 1) / fps_float) * 1000.0
                        detections = []
                        for det in results.boxes.data.cpu().numpy():
                            x1, y1, x2, y2, conf, cls = det
                            if int(cls) in [2, 5, 7]:  # car, bus, truck
                                w, h = x2 - x1, y2 - y1
                                detections.append(((x1, y1, w, h), int(cls)))
                        
                        # Draw all detections first (bounding boxes and labels)
                        for (bbox, cls) in detections:
                            scaled_bbox = scale_bbox(bbox, scale, offset_x, offset_y)
                            x, y, w, h = scaled_bbox
                            label = model_instance.names[cls]
                            cv2.rectangle(output_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                            cv2.putText(output_frame, label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Draw distance lines and measurements for pairs under threshold
                        if len(detections) >= 2:
                            for i in range(len(detections)):
                                for j in range(i + 1, len(detections)):
                                    (bbox1, cls1), (bbox2, cls2) = detections[i], detections[j]
                                    distance = compute_distance_with_homography(bbox1, bbox2, h_matrix, METER_PER_PIXEL)
                                    did_violate = "Yes" if distance < DISTANCE_THRESHOLD_METERS else "No"

                                    # Log violation status for each detection pair
                                    log_report_entry(csv_writer2, frame_timestamp_ms, did_violate)

                                    # Draw distance line and measurement only if under threshold
                                    if did_violate == "Yes":
                                        # Scale bounding box coordinates for target resolution
                                        scaled_bbox1 = scale_bbox(bbox1, scale, offset_x, offset_y)
                                        scaled_bbox2 = scale_bbox(bbox2, scale, offset_x, offset_y)
                                        
                                        cx1, cy1 = int(scaled_bbox1[0] + scaled_bbox1[2] / 2), int(scaled_bbox1[1] + scaled_bbox1[3] / 2)
                                        cx2, cy2 = int(scaled_bbox2[0] + scaled_bbox2[2] / 2), int(scaled_bbox2[1] + scaled_bbox2[3] / 2)
                                        
                                        cv2.line(output_frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
                                        mid_x, mid_y = int((cx1 + cx2) / 2), int((cy1 + cy2) / 2)
                                        cv2.putText(output_frame, f"{distance:.2f} m", (mid_x, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Write frame to output video
                        out.write(output_frame)
                        
                        # Log progress every 50 frames
                        if idx % 50 == 0:
                            logging.info(f"FFmpeg fallback processed {idx}/{extracted} frames")
                
                total_good_frames = extracted
                logging.info(f"FFmpeg fallback completed: processed {extracted} frames with full annotations")
            else:
                logging.error("FFmpeg frame extraction produced 0 frames; keeping existing output")
        
        logging.info(f"Video processing finished for {video_blob_name}.")
        cap.release()
        out.release()
        
        # Convert processed MP4 back to AVI for final output
        final_avi_output = os.path.join(temp_dir, f"processed_{video_name_without_ext}.avi")
        
        if temp_mp4_output.endswith('.mp4'):
            logging.info("🔄 Converting processed MP4 back to AVI for final output...")
            if convert_mp4_to_avi(temp_mp4_output, final_avi_output):
                logging.info("✅ Successfully converted processed video back to AVI")
                final_output_path = final_avi_output
            else:
                logging.warning("⚠️ Failed to convert to AVI, using MP4 output")
                final_output_path = temp_mp4_output
        else:
            # Already AVI format
            final_output_path = temp_mp4_output
        
        original_video_base = os.path.splitext(os.path.basename(video_blob_name))[0]
        
        # Use the improved upload function with better folder structure
        logging.info(f"☁️  Uploading processed results to GCS...")
        upload_start = time.time()
        upload_results_to_gcs(DESTINATION_BUCKET_NAME, final_output_path, csv_report_temp_path, original_video_base)
        upload_time = time.time() - upload_start
        logging.info(f"☁️  Upload completed in {upload_time:.2f} seconds")
        
        # Calculate and log processing time
        end_time = time.time()
        processing_duration = end_time - start_time
        hours = int(processing_duration // 3600)
        minutes = int((processing_duration % 3600) // 60)
        seconds = int(processing_duration % 60)
        
        # Calculate video quality metrics
        total_frames_processed = total_good_frames + total_bad_frames
        quality_score = (total_good_frames / total_frames_processed) * 100 if total_frames_processed > 0 else 0
        
        logging.info(f"⏱️  PROCESSING COMPLETE!")
        logging.info(f"📊 Total processing time: {hours:02d}:{minutes:02d}:{seconds:02d} ({processing_duration:.2f} seconds)")
        logging.info(f"🎬 Video: {video_blob_name}")
        logging.info(f"📈 Frames processed: {total_good_frames} good, {total_bad_frames} bad")
        logging.info(f"🎯 Video quality score: {quality_score:.1f}% (higher is better)")
        
        # Quality assessment
        if quality_score >= 90:
            logging.info(f"✅ Excellent video quality - minimal corruption")
        elif quality_score >= 70:
            logging.info(f"⚠️  Good video quality - some corruption handled")
        elif quality_score >= 50:
            logging.info(f"⚠️  Fair video quality - significant corruption but processed")
        else:
            logging.info(f"❌ Poor video quality - heavily corrupted but processed with available frames")
        
        # Breakdown of processing time
        logging.info(f"⏱️  TIME BREAKDOWN:")
        logging.info(f"   📥 Download: {download_time:.2f}s ({download_time/processing_duration*100:.1f}%)")
        if input_was_converted:
            logging.info(f"   🔄 Conversion: {conversion_time:.2f}s ({conversion_time/processing_duration*100:.1f}%)")
        logging.info(f"   🎬 Frame processing: {frame_processing_time:.2f}s ({frame_processing_time/processing_duration*100:.1f}%)")
        logging.info(f"   ☁️  Upload: {upload_time:.2f}s ({upload_time/processing_duration*100:.1f}%)")
        
        if total_good_frames > 0:
            fps_actual = total_good_frames / processing_duration
            logging.info(f"⚡ Overall processing speed: {fps_actual:.2f} frames/second")

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


# Processing status file functions removed - using in-memory tracking only

def load_models():
    """Loads models into global variables at startup."""
    global model, H
    try:
        logging.info("Loading YOLO model and homography matrix...")
        
        # Models should be in the current directory
        model_local_path = YOLO_MODEL_FILENAME
        homography_local_path = HOMOGRAPHY_FILENAME

        if not os.path.exists(model_local_path):
            raise FileNotFoundError(f"YOLO model file not found: {model_local_path}")
        if not os.path.exists(homography_local_path):
            raise FileNotFoundError(f"Homography file not found: {homography_local_path}")

        # Setup GPU first
        gpu_available = setup_gpu()
        
        logging.info("Loading YOLO model...")
        model = YOLO(model_local_path)
        
        # Move model to GPU if available
        if gpu_available:
            model.to(device)
            logging.info(f"✅ YOLO model moved to {device}")
        else:
            logging.info("⚠️  YOLO model running on CPU")
        
        logging.info("Loading homography matrix...")
        H = np.load(homography_local_path)
        
        # Log device information
        device_info = get_device_info()
        logging.info(f"🎯 Running on: {device_info}")
        
        logging.info("Models loaded successfully.")
        
    except Exception as e:
        logging.error(f"FATAL: Failed to load models on startup. Error: {e}")
        raise

# Orphaned processing files cleanup removed - using in-memory tracking only

def start_pubsub_worker():
    """Start the Pub/Sub worker."""
    logging.info("--- Starting Pub/Sub Pull Worker ---")
    
    # Load models
    load_models()

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

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

            # Process videos from any folder in the bucket
            logging.info(f"Processing video from path: {video_blob_name}")
            
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

                # Note: Using in-memory tracking only for processed videos
                logging.info(f"Using in-memory tracking for {video_blob_name}")

                # Note: Cross-instance duplicate prevention removed - using in-memory tracking only

                # Process the video
                try:
                    logging.info(f"🚀 Starting video processing: {video_blob_name}")
                    process_video(video_blob_name, model, H)
                    
                    # Move from processing to processed
                    with processing_lock:
                        processing_videos.discard(video_blob_name)
                        processed_videos.add(video_blob_name)
                        
                        # Clean up old entries to prevent memory issues
                        if len(processed_videos) > MAX_PROCESSED_VIDEOS:
                            # Remove oldest entries (keep last 500)
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

def main():
    """The main entry point for the Compute Engine worker."""
    logging.info("--- Starting YOLO Range Detection Worker on Compute Engine ---")
    
    # Log GPU information at startup
    if torch.cuda.is_available():
        logging.info("🚀 CUDA is available - GPU acceleration enabled!")
        logging.info(f"📊 CUDA Version: {torch.version.cuda}")
        logging.info(f"🔥 PyTorch Version: {torch.__version__}")
    else:
        logging.warning("⚠️  CUDA not available - running on CPU")
    
    # Start the Pub/Sub worker
    start_pubsub_worker()

if __name__ == "__main__":
    main()
