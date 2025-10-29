"""
Core contraflow detection algorithm - NUMPY OPTIMIZED.
"""
import cv2
import numpy as np
import logging

# Enable numpy optimizations for maximum performance
np.seterr(divide='ignore', invalid='ignore')  # Suppress warnings for speed
from .config import (
    ROIS,
    EXPECTED_DIRECTIONS,
    ROI_CENTER_INTERSECTION,
    CONTRAFLOW_THRESHOLD,
    WRONG_WAY_DISTANCE_THRESHOLD,
    LANE_COLORS,
    CONTRAFLOW_COLOR,
    OK_COLOR,
)

logger = logging.getLogger(__name__)


class ContraflowDetector:
    """Core contraflow detection logic."""
    
    def __init__(self):
        # ROI definitions
        self.rois = {
            1: np.array([[1132, 354], [862, 578], [280, 102], [418, 97]], np.int32),  # Blue Lane
            2: np.array([[412, 90], [574, 63], [1246, 283], [1132, 354]], np.int32),  # Green Lane
            3: np.array([[1189, 117], [1034, 84], [116, 242], [426, 552]], np.int32), # Yellow Lane
            4: np.array([[67, 161], [116, 242], [1030, 87], [940, 65]], np.int32)     # Red Lane
        }
        self.roi_center_intersection = np.array([[326, 140], [665, 417], [1025, 211], [679, 96]], np.int32)
        
        # Constants
        self.TRACKING_POINT_Y_OFFSET = 0.23  # 80% down from center
        self.MINIMUM_MOVEMENT_THRESHOLD = 5
        self.DIRECTION_VIOLATION_CONFIRMATION_FRAMES = 10
        self.PATH_VIOLATION_CONFIRMATION_FRAMES = 2
        self.wrong_way_distance_threshold = float(WRONG_WAY_DISTANCE_THRESHOLD)
        self.contraflow_threshold = float(CONTRAFLOW_THRESHOLD)
        
        # Legal directions
        self.legal_directions = {
            1: np.array([-0.707, -0.707]),  # Blue Lane
            2: np.array([0.707, 0.707]),    # Green Lane 
            3: np.array([-0.707, 0.707]),   # Yellow Lane
            4: np.array([0.707, -0.707])    # Red Lane
        }
        
        # Opposing lanes
        self.opposite_lanes = {1: 2, 2: 1, 3: 4, 4: 3}
        
        # Tracking state
        self.reset()

    def reset(self):
        """Reset tracking state for new video"""
        self.track_history = {}
        self.vehicle_assigned_lane = {}
        self.vehicle_direction_violation_counter = {}
        self.vehicle_path_violation_counter = {}
        self.logged_contraflow_ids = set()
        self.contraflow_screenshot_taken = set()
        self.vehicle_wrong_way_distance = {}
    
    def get_direction(self, point1, point2):
        """Calculate the direction vector between two points - OPTIMIZED."""
        # Use float32 for faster operations and direct calculation
        dx = np.float32(point2[0] - point1[0])
        dy = np.float32(point2[1] - point1[1])
        
        # Fast norm calculation using rsqrt approximation
        norm_squared = dx * dx + dy * dy
        if norm_squared > 1e-8:  # Avoid division by zero with threshold
            inv_norm = np.float32(1.0) / np.sqrt(norm_squared)
            return np.array([dx * inv_norm, dy * inv_norm], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)
    
    def assign_vehicle_to_lane(self, track_id, center_point):
        """Assign vehicle to a lane based on its position."""
        if track_id not in self.vehicle_assigned_lane:
            for lane_id, roi in ROIS.items():
                if cv2.pointPolygonTest(roi, center_point, False) >= 0:
                    self.vehicle_assigned_lane[track_id] = lane_id
                    logger.debug(f"Assigned vehicle {track_id} to lane {lane_id}")
                    break
    
    def is_contraflow(self, track_id, center_point):
        """
        Detect if a vehicle is moving in contraflow (wrong direction).
        
        Args:
            track_id: Unique identifier for the tracked vehicle
            center_point: Current center position of the vehicle
            
        Returns:
            bool: True if contraflow detected, False otherwise
        """
        # Assign vehicle to lane if not already assigned
        self.assign_vehicle_to_lane(track_id, center_point)
        
        assigned_lane = self.vehicle_assigned_lane.get(track_id)
        current_distance = self.vehicle_wrong_way_distance.get(track_id, 0.0)

        # Need lane assignment and sufficient tracking history
        if not assigned_lane or len(self.track_history[track_id]) <= 5:
            self.vehicle_wrong_way_distance[track_id] = current_distance
            return False
        
        # Check if vehicle is in the center intersection dead zone
        in_center_zone = cv2.pointPolygonTest(ROI_CENTER_INTERSECTION, center_point, False) >= 0
        
        # Don't apply contraflow checks in the center dead zone
        if in_center_zone:
            self.vehicle_wrong_way_distance[track_id] = 0.0
            return False
        
        # Calculate current movement direction
        last_point = self.track_history[track_id][-1]
        prev_point = self.track_history[track_id][-5]
        current_direction = self.get_direction(prev_point, last_point)
        
        # Get list of expected directions for the lane
        expected_directions = EXPECTED_DIRECTIONS.get(assigned_lane)
        if expected_directions:
            direction_scores = [float(np.dot(current_direction, exp_dir)) for exp_dir in expected_directions]
            if direction_scores:
                max_direction_score = max(direction_scores)
            else:
                max_direction_score = -1.0

            segment_distance = float(
                np.linalg.norm(np.array(last_point, dtype=np.float32) - np.array(prev_point, dtype=np.float32))
            )
            if not np.isfinite(segment_distance):
                segment_distance = 0.0

            if max_direction_score > self.contraflow_threshold:
                updated_distance = 0.0
            else:
                updated_distance = current_distance + segment_distance

            self.vehicle_wrong_way_distance[track_id] = updated_distance

            is_contraflow = (
                max_direction_score <= self.contraflow_threshold
                and updated_distance >= self.wrong_way_distance_threshold
            )
            if is_contraflow and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Contraflow detected for track %s: score %.3f (threshold %.3f), distance %.2f/%.2f",
                    track_id,
                    max_direction_score,
                    self.contraflow_threshold,
                    updated_distance,
                    self.wrong_way_distance_threshold,
                )
            return is_contraflow
        
        self.vehicle_wrong_way_distance[track_id] = current_distance
        return False
    
    def update_track_history(self, track_id, center_point):
        """Update tracking history for a vehicle."""
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(center_point)
    
    def should_log_contraflow(self, track_id):
        """Check if contraflow event should be logged (first time only)."""
        if track_id not in self.logged_contraflow_ids:
            self.logged_contraflow_ids.add(track_id)
            return True
        return False
    
    def should_take_screenshot(self, track_id):
        """Check if screenshot should be taken (first time only)."""
        if track_id not in self.contraflow_screenshot_taken:
            self.contraflow_screenshot_taken.add(track_id)
            return True
        return False
    
    def draw_lanes_on_frame(self, frame):
        """Draw lane ROIs and center intersection on frame."""
        # Draw lane ROIs with colors
        cv2.polylines(frame, [ROIS[1]], isClosed=True, color=LANE_COLORS[1], thickness=2)  # Blue
        cv2.polylines(frame, [ROIS[2]], isClosed=True, color=LANE_COLORS[2], thickness=2)  # Green
        cv2.polylines(frame, [ROIS[3]], isClosed=True, color=LANE_COLORS[3], thickness=2)  # Yellow
        cv2.polylines(frame, [ROIS[4]], isClosed=True, color=LANE_COLORS[4], thickness=2)  # Magenta
        
        # Draw center intersection dead zone
        cv2.polylines(frame, [ROI_CENTER_INTERSECTION], isClosed=True, 
                     color=LANE_COLORS['center'], thickness=2)  # White
        
        return frame

    
    def draw_track_history(self, frame, track_id):
        """Draw vehicle tracking trail on frame - OPTIMIZED."""
        if track_id in self.track_history and len(self.track_history[track_id]) > 1:
            # Pre-allocate array with correct dtype for faster operations
            points_list = self.track_history[track_id]
            points = np.empty((len(points_list), 1, 2), dtype=np.int32)
            for i, point in enumerate(points_list):
                points[i, 0] = [int(point[0]), int(point[1])]
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=1)
        return frame
    
    def annotate_vehicle(self, frame, track_id, x1, y1, x2, y2, is_contraflow_detected):
        """
        Annotate vehicle on frame with bounding box and status.
        
        Args:
            frame: Video frame to annotate
            track_id: Vehicle tracking ID
            x1, y1, x2, y2: Bounding box coordinates
            is_contraflow_detected: Boolean indicating contraflow status
            
        Returns:
            Annotated frame
        """
        # Determine colors and labels
        if is_contraflow_detected:
            box_color = CONTRAFLOW_COLOR  # Red for contraflow
            status_text = "Contraflow"
        else:
            box_color = OK_COLOR  # Green for OK
            status_text = "OK"
        
        # Get lane assignment for display
        assigned_lane = self.vehicle_assigned_lane.get(track_id)
        lane_info = f"L{assigned_lane}" if assigned_lane else "Unknown"
        
        # Create label with vehicle info
        label = f"#{track_id} {lane_info} {status_text}"
        
        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        text_color = (255, 255, 255)
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw label background
        label_bg_x1 = x1
        label_bg_y1 = y1 - 10 - text_height - baseline
        label_bg_x2 = x1 + text_width
        label_bg_y2 = y1 - 10
        
        cv2.rectangle(frame, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), box_color, cv2.FILLED)
        
        # Draw text label
        cv2.putText(frame, label, (x1, y1 - 10 - baseline), font, font_scale, text_color, font_thickness)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        return frame
