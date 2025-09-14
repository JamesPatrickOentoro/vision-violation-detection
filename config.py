import cv2
import supervision as sv
from inference import get_model
from tqdm import tqdm
import matplotlib.pyplot as plt
from roboflow import Roboflow
import numpy as np
import os

ROAD = np.array([[492, 236], [1047, 716], [1274, 714], [1276, 442], [747, 167]])

STOP_ZONE = np.array([[876, 566], [1131, 401], [1277, 490], [1274, 707], [1258, 717], [1017, 717]])

OUT_ZONE = np.array([[876, 563], [1126, 396], [820, 235], [609, 310]])

POINT= (1062, 505)