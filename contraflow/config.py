"""
Configuration constants for contraflow detection system.
"""
import numpy as np

# Google Cloud Configuration - Use environment variables
import os
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    current_file = Path(__file__)
    print(f"🔍 Config file location: {current_file.absolute()}")
    config_dir = current_file.parent.parent.parent
    print(f"🔍 Calculated project root: {config_dir.absolute()}")
    env_path = config_dir / '.env'
    print(f"🔍 Looking for .env at: {env_path.absolute()}")
    print(f"🔍 .env exists: {env_path.exists()}")
    
    if env_path.exists():
        result = load_dotenv(env_path)
        print(f"🔧 Config: load_dotenv result: {result}")
    else:
        print(f"⚠️ Config: No .env file found")
except ImportError as e:
    print(f"❌ Config: python-dotenv not installed ({e})")
    print("💡 Install with: pip install python-dotenv")

PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'your-project-id')

# Hardcoded identifiers for runtime (overrides any environment variables)
SUBSCRIPTION_NAME = 'adaro-contraflow-detection-sub'
BUCKET_NAME = 'vision-poc-bucket-adaro'
# Separate destination bucket for reports (and optionally screenshots)
REPORTS_BUCKET_NAME = 'adaro_vision_contraflow'

# Processing Configuration
PULL_TIMEOUT = 30  # Seconds to wait for messages
MAX_MESSAGES = 1   # Process one video at a time
RETRY_DELAY = 5    # Seconds to wait before retrying on error

# Optimized Pub/Sub settings for maximum throughput
ACK_DEADLINE_EXTENSION_INTERVAL = 60  # Seconds between ack deadline extensions
MAX_ACK_DEADLINE_EXTENSION = 1200     # Reduced to 20 minutes for faster failure detection
ESTIMATED_PROCESSING_TIME = 400       # Optimistic estimate with direct encoding (was 600)
MEMORY_CLEANUP_INTERVAL = 50          # More frequent cleanup for small VMs
MAX_PROCESSED_VIDEOS_MEMORY = 500     # Reduced memory usage (removed anyway in optimization)

# Video Processing Configuration
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Model Configuration - TensorRT Optimized (Local Models Directory)
import os
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = str(PROJECT_ROOT / 'models')

MODEL_CLASSES = [2, 3, 5, 7]  # Vehicle classes for YOLO detection

# TensorRT Model Paths (prioritized for maximum performance)
TENSORRT_MODEL_PATHS = {
    'nano': f'{MODELS_DIR}/yolo11n.engine',     # TensorRT INT8 - Fastest
    'small': f'{MODELS_DIR}/yolo11s.engine',    # TensorRT INT8 - Balanced
}

# PyTorch Model Paths (fallback)
PYTORCH_MODEL_PATHS = {
    'nano': f'{MODELS_DIR}/yolo11n.pt',         # PyTorch - Fallback
    'small': f'{MODELS_DIR}/yolo11s.pt',        # PyTorch - Fallback
}

# Default model selection (will auto-detect TensorRT vs PyTorch)
DEFAULT_MODEL_SIZE = 'nano'  # Options: 'nano', 'small'

# Legacy compatibility
MODEL_PATH = TENSORRT_MODEL_PATHS[DEFAULT_MODEL_SIZE]

# Lane Regions of Interest (ROIs) for 1280x720 resolution
# Lane 1: Bottom-right to Top-left (where the large trucks are)
ROI_LANE1 = np.array([[1132, 354], [862, 578], [280, 102], [412, 90]], np.int32)

# Lane 2: Top-left to Bottom-right (opposite of lane 1)
ROI_LANE2 = np.array([[412, 90], [574, 63], [1246, 283], [1132, 354]], np.int32)

# Lane 3: Top-right to Bottom-left
ROI_LANE3 = np.array([[1189, 117], [1034, 84], [116, 242], [426, 552]], np.int32)

# Lane 4: Bottom-left to Top-right (where the white pickup is heading)
ROI_LANE4 = np.array([[61, 151], [116, 242], [1030, 87], [940, 65]], np.int32)

# Center intersection dead zone
ROI_CENTER_INTERSECTION = np.array([[308, 125], [665, 417], [1025, 211], [664, 91]], np.int32)

# Store all ROIs in a dictionary
ROIS = {
    1: ROI_LANE1,
    2: ROI_LANE2, 
    3: ROI_LANE3,
    4: ROI_LANE4
}

# Opposing lanes mapping
OPPOSING_LANES = {
    1: 2,
    2: 1,
    3: 4,
    4: 3 
}

# Detection thresholds - Optimized for numpy float32
CONTRAFLOW_THRESHOLD = np.float32(-0.3)
DIRECTION_THRESHOLD = np.float32(0.1)
WRONG_WAY_DISTANCE_THRESHOLD = np.float32(
    float(os.getenv("WRONG_WAY_DISTANCE_THRESHOLD", "60.0"))
)

# Expected traffic directions for each lane - Pre-normalized for maximum performance
EXPECTED_DIRECTIONS = {
    1: [np.array([-0.894, -0.447], dtype=np.float32), np.array([0.894, -0.447], dtype=np.float32)], # Lane 1 (Blue): Straight-left OR turn right
    2: [np.array([0.894, 0.447], dtype=np.float32)],                                                # Lane 2 (Green): Straight-right
    3: [np.array([-0.894, 0.447], dtype=np.float32)],                                               # Lane 3 (Yellow): Straight-left
    4: [np.array([0.894, -0.447], dtype=np.float32)]                                                # Lane 4 (Magenta): Straight-right
}

# Lane colors for visualization (BGR format for OpenCV)
LANE_COLORS = {
    1: (255, 0, 0),    # Blue
    2: (0, 255, 0),    # Green
    3: (0, 255, 255),  # Yellow
    4: (0, 0, 255),    # Magenta
    'center': (255, 255, 255)  # White for center dead zone
}

# Detection colors
CONTRAFLOW_COLOR = (0, 0, 255)  # Red for contraflow
OK_COLOR = (0, 255, 0)          # Green for OK

# Optimized processing limits for maximum performance
MAX_CONSECUTIVE_FAILURES = 3  # Fail faster to avoid wasting resources
FRAME_BUFFER = 20  # Reduced buffer for faster recovery
PROGRESS_LOG_INTERVAL = 1000  # Reduced logging frequency

# Aggressive memory management for maximum utilization
GPU_MEMORY_FRACTION = 0.95  # Use 95% GPU memory (leave room for TensorRT)
CPU_GPU_MEMORY_FRACTION = 0.85  # Use more CPU memory when no GPU

# TensorRT Optimization Settings for Tesla T4
TENSORRT_ENABLED = True
TENSORRT_BATCH_SIZE = 1  # Single batch for consistent latency
TENSORRT_WORKSPACE_SIZE = 8  # GB - Tesla T4 has 16GB VRAM
TENSORRT_INT8_CALIBRATION = True  # Enable INT8 quantization

# GPU Utilization Optimization Settings
FRAME_SKIP = 1  # Process every frame (no skipping for maximum detection accuracy)

# Parallel Processing Settings
PARALLEL_WORKERS = 4  # Number of parallel processing threads
FRAME_QUEUE_SIZE = 16  # Buffer size for frame processing queue

# File paths (using project-relative directories)
SCREENSHOTS_DIR = str(PROJECT_ROOT / "screenshots")
TEMP_FRAMES_DIR = "/tmp"
