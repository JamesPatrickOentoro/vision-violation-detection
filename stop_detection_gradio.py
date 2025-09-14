from google.cloud import storage
from google.oauth2 import service_account
import supervision as sv
from config import *
import tempfile
import cv2
from inference import get_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from roboflow import Roboflow
import numpy as np
import os
from stopvideo import StopVideo

SERVICE_ACCOUNT_KEY_PATH = "adaro-vision-poc-e36a1eaf86f5.json"
GCS_BUCKET_NAME = "vision-poc-bucket-adaro"
your_video_blob_name = "2025-07-22/14/2025-07-22 14-00-00~14-05-03.avi" # The exact object name
local_download_path = "downloaded_video_from_sa_key.avi"
first_video = StopVideo('video2',SERVICE_ACCOUNT_KEY_PATH,GCS_BUCKET_NAME,your_video_blob_name,'video2.mp4')
first_video.inference()
first_video.analyze()
first_video.render_video_advanced()