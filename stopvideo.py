import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# --- Constants ---
# TODO: Fill in the correct polygon coordinates for your video source.
# These values are placeholders and must be adjusted for your specific camera setup.
STOP_ZONE = np.array([
    [876, 566], [1131, 401], [1277, 490], [1274, 707], [1258, 717], [1017, 717]
])
OUT_ZONE = np.array([
    [876, 563], [1126, 396], [820, 235], [609, 310]
])
ROAD = np.array([
    [492, 236], [1047, 716], [1274, 714], [1276, 442], [747, 167]
])
POINT = (1062, 505)


class StopVideo:
    """
    A class to process a video, detect vehicles and their wheels,
    and analyze if they stop at a designated stop zone.
    It uses YOLOv8 for vehicle detection and a custom YOLO model for wheel detection.
    """
    def __init__(self, video_name, local_video_path, wheel_model_path):
        """
        Initializes the StopVideo processor.

        Args:
            video_name (str): A name for the video being processed, used for output directories.
            local_video_path (str): The file path to the local video.
            wheel_model_path (str): The file path to the custom-trained wheel detection model (.pt file).
        """
        self.video_name = video_name
        self.output_path = f"output/{video_name}"
        os.makedirs(self.output_path, exist_ok=True)
        
        self.local_video_path = local_video_path
        
        # Load video data
        self.video_info, self.frames_generator = self.load_video_local()
        
        # --- Model Loading ---
        # Load the custom wheel detection model from the provided path.
        # This model should be downloaded from the GitHub repository releases.
        print(f"Loading wheel detection model from: {wheel_model_path}")
        self.model_wheels = YOLO(wheel_model_path)
        
        # Load a general-purpose, pre-trained YOLOv8 model for vehicle detection.
        # The model will be downloaded automatically on first use.
        print("Loading vehicle detection model (yolov11s.pt)...")
        self.model_yolo = YOLO('yolov11s.pt') # Corrected from yolo11s.pt to a standard model
    
        # --- Trackers ---
        # Initialize ByteTrack for tracking both vehicles and wheels.
        self.tracker_yolo = sv.ByteTrack(frame_rate=self.video_info.fps)
        self.tracker_wheel = sv.ByteTrack(frame_rate=self.video_info.fps)

        # --- Annotators ---
        # Setup annotators for drawing boxes and labels with different colors.
        self.box_annotator_green = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
        self.box_annotator_gray = sv.BoxAnnotator(color=sv.Color(r=176, g=178, b=181), thickness=2)
        self.box_annotator_red = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
        self.box_annotator_white = sv.BoxAnnotator(color=sv.Color.WHITE, thickness=2)

        scale = 0.5
        self.label_annotator_green = sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=scale, text_thickness=1)
        self.label_annotator_gray = sv.LabelAnnotator(color=sv.Color(r=176, g=178, b=181), text_scale=scale, text_thickness=1)
        self.label_annotator_red = sv.LabelAnnotator(color=sv.Color.RED, text_scale=scale, text_thickness=1)

        # --- Zones ---
        # Define and annotate the polygonal zones for stop, out, and road areas.
        self.stopzone, self.stopzone_annotator = self.add_zone(sv.Color.GREEN, STOP_ZONE)
        self.outzone, self.outzone_annotator = self.add_zone(sv.Color.RED, OUT_ZONE)   
        self.road, self.road_annotator = self.add_zone(sv.Color.BLUE, ROAD)             

        self.point = POINT

    def inference(self):
        """
        Performs inference frame-by-frame to detect and track cars and wheels.
        """
        # Dictionaries to store detection and tracking data per frame
        self.car_detections = {}
        self.wheel_detections = {}
        self.report = {}
        self.car_center_history = {}
        self.first_last_frames = {}

        # Use supervision's frame generator for robust video processing
        frames_generator = sv.get_video_frames_generator(source_path=self.local_video_path)
        
        for frame_no, frame in enumerate(tqdm(frames_generator, total=self.video_info.total_frames, desc="Processing Frames")):
            
            # --- 1. Vehicle Detection ---
            results_yolo = self.model_yolo(frame, verbose=False)[0]
            detections_yolo_all = sv.Detections.from_ultralytics(results_yolo)
            
            detections_yolo_vehicle = detections_yolo_all[self.is_vehicle(detections_yolo_all)]
            detections_yolo_vehicle = self.tracker_yolo.update_with_detections(detections_yolo_vehicle)

            car_ids = detections_yolo_vehicle.tracker_id
            for id in car_ids:
                if id not in self.first_last_frames:
                    self.first_last_frames[id] = {"first": frame_no, "last": frame_no}
                else:
                    self.first_last_frames[id]["last"] = frame_no
            
            on_road_mask = self.road.trigger(detections=detections_yolo_vehicle)
            detections_yolo_vehicle = detections_yolo_vehicle[on_road_mask]
            detections_yolo_vehicle = self.return_target(detections_yolo_vehicle)
            
            vehicle, vehicle_coords = self.extract_vehicle(frame, detections_yolo_vehicle)
            
            if len(detections_yolo_vehicle) > 0:
                self.car_detections[frame_no] = detections_yolo_vehicle

            # --- 2. Wheel Detection ---
            if vehicle is not None:
                results_wheel = self.model_wheels(vehicle, verbose=False)[0]
                detections_wheel = sv.Detections.from_ultralytics(results_wheel)

                if len(detections_wheel) > 0:
                    detections_wheel = self.update_wheel_coords(detections_wheel, vehicle_coords)
                
                detections_wheel = self.tracker_wheel.update_with_detections(detections_wheel)
                
                if len(detections_wheel) > 0:
                    self.wheel_detections[frame_no] = detections_wheel

            # --- 3. Reporting ---
            self.report[frame_no] = self.construct_report(frame_no)
            
        print("Inference complete.")
        return self.report

    def is_vehicle(self, detections):
        """Filters detections to only include COCO vehicle classes."""
        # COCO class IDs for vehicles: 2:car, 3:motorcycle, 5:bus, 7:truck
        vehicle_class_ids = [2, 3, 5, 7]
        return np.isin(detections.class_id, vehicle_class_ids)
    
    def get_ids(self, detections):
        """Returns the tracker_id of the first detection."""
        if detections is None or len(detections) == 0:
            return None
        return detections.tracker_id[0]
    
    def return_target(self, detections):
        """Filters detections to return only the one closest to the POI."""
        if len(detections) == 0:
            return detections
        
        distances = [self.get_distance_to_point(xyxy) for xyxy in detections.xyxy]
        min_dist_index = np.argmin(distances)
        # FIX: Use list-based indexing to preserve the 2D shape of the array
        return detections[[min_dist_index]]
        
    def get_distance_to_point(self, xyxy):
        """Calculates Euclidean distance from a bounding box center to the POI."""
        x1, y1, x2, y2 = xyxy
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        return np.linalg.norm(center - np.array(self.point))
        
    def extract_vehicle(self, frame, detections):
        """Crops the vehicle from the frame based on its bounding box."""
        if len(detections) == 0:
            return None, None
        
        # We assume return_target has filtered to one detection
        x1, y1, x2, y2 = detections.xyxy[0].astype(int)
        # Add padding to ensure the whole vehicle is captured
        y1 = max(0, y1 - 20)
        y2 = min(frame.shape[0], y2 + 20)
        x1 = max(0, x1 - 20)
        x2 = min(frame.shape[1], x2 + 20)
        vehicle = frame[y1:y2, x1:x2]
        vehicle_coords = (x1, y1, x2, y2)
        return vehicle, vehicle_coords

    def update_wheel_coords(self, wheel_detections, vehicle_coords):
        """Adjusts wheel bounding box coordinates from the vehicle crop to the full frame."""
        x1, y1, _, _ = vehicle_coords
        wheel_detections.xyxy += np.array([x1, y1, x1, y1])
        return wheel_detections

    def construct_report(self, frame_no):
        """Constructs a per-frame summary of car and wheel status."""
        if frame_no not in self.car_detections:
            return {}
        
        car_id = self.get_ids(self.car_detections[frame_no])
        if car_id is None:
            return {}

        frame_report = {car_id: {}}
        
        if frame_no in self.wheel_detections:
            wheels = self.wheel_detections[frame_no]
            wheel_ids = wheels.tracker_id
            in_stopzone = self.stopzone.trigger(detections=wheels)
            in_outzone = self.outzone.trigger(detections=wheels)
            
            wheel_list = [
                {wheel_id: {"stopzone": stop, "outzone": out}}
                for wheel_id, stop, out in zip(wheel_ids, in_stopzone, in_outzone)
            ]
            frame_report[car_id]["wheel"] = wheel_list

        car_center = self.get_car_center(frame_no)
        frame_report[car_id]["car_center"] = car_center
        
        # Calculate speed
        if car_id not in self.car_center_history:
            self.car_center_history[car_id] = [None, car_center]
        else:
            self.car_center_history[car_id].append(car_center)
        
        prev_center = self.car_center_history[car_id][-2]
        if prev_center is not None and car_center is not None:
            distance_moved = np.linalg.norm(np.array(car_center) - np.array(prev_center))
            current_speed = distance_moved * self.video_info.fps
        else:
            current_speed = None
            
        frame_report[car_id]["speed"] = current_speed
        return frame_report
    
    def get_car_center(self, frame_no):
        """Returns the center of the car's bounding box."""
        if frame_no not in self.car_detections or len(self.car_detections[frame_no].xyxy) == 0:
            return None
        x1, y1, x2, y2 = self.car_detections[frame_no].xyxy[0]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def load_video_local(self):
        """Loads video info and creates a frame generator."""
        print(f"Loading video from local path: '{self.local_video_path}'")
        try:
            video_info = sv.VideoInfo.from_video_path(video_path=self.local_video_path)
            # The frames_generator is created in the inference method now
            # using sv.get_video_frames_generator for better compatibility.
            return video_info, None 
        except Exception as e:
            print(f"An error occurred in load_video_local: {e}")
            raise

    def add_zone(self, color, zone_coords):
        """Creates a PolygonZone and its corresponding annotator."""
        zone = sv.PolygonZone(polygon=zone_coords)
        zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=color)
        return zone, zone_annotator

    def analyze(self):
        """Analyzes the full report to determine if each car stopped."""
        wheels_in_stopzone = {}
        wheels_in_outzone = {}
        self.car_left_zone = {}
        self.car_speeds = {}
        stopped = {}
        car_id_history = []
        
        self.reported_wheels = {}
        self.analysis = {}
        
        for frame_id, frame_dict in self.report.items():
            for car_id, vehicle_dict in frame_dict.items():
                if car_id not in car_id_history:
                    car_id_history.append(car_id)

                if not vehicle_dict or "wheel" not in vehicle_dict:
                    continue
                
                if car_id not in wheels_in_stopzone: wheels_in_stopzone[car_id] = []
                if car_id not in wheels_in_outzone: wheels_in_outzone[car_id] = []

                for wheel_dict in vehicle_dict["wheel"]:
                    for wheel_id, wheel_info in wheel_dict.items():
                        if wheel_info["stopzone"] and wheel_id not in wheels_in_stopzone[car_id]:
                            wheels_in_stopzone[car_id].append(wheel_id)
                        if wheel_info["outzone"] and wheel_id not in wheels_in_outzone[car_id]:
                            wheels_in_outzone[car_id].append(wheel_id)
                
                if len(wheels_in_stopzone.get(car_id, [])) > 0:
                    if car_id not in self.car_speeds: self.car_speeds[car_id] = []
                    self.car_speeds[car_id].append({"frame_no": frame_id, "speed": vehicle_dict.get('speed')})
                
                for wheel in wheels_in_stopzone.get(car_id, []):
                    if wheel in wheels_in_outzone.get(car_id, []):
                        if car_id not in self.reported_wheels:
                            self.reported_wheels[car_id] = wheel
                        if car_id not in stopped:
                            speeds = self.car_speeds.get(car_id, [])
                            filtered_speeds = [s for s in speeds if s["speed"] is not None]
                            if not filtered_speeds:
                                stopped[car_id] = False
                                self.car_left_zone[car_id] = frame_id
                                continue

                            min_speed_in_stopzone = min(filtered_speeds, key=lambda x: x['speed'])
                            # Speed threshold (pixels/sec). Adjust as needed.
                            if min_speed_in_stopzone["speed"] < 15:
                                stopped[car_id] = True
                            else:
                                stopped[car_id] = False
                            self.car_left_zone[car_id] = min_speed_in_stopzone["frame_no"]
        
        for car_id in car_id_history:
            if car_id in self.reported_wheels:
                status = "Stopped" if stopped.get(car_id) else "Failed to stop"
                self.analysis[car_id] = {
                    "First Entrance": self.first_last_frames[car_id]["first"],
                    "Last Exit": self.first_last_frames[car_id]["last"],
                    "Status": status
                }
        print("Analysis complete.")

    def render_video_simple(self):
        """Renders a simple video showing only the final status of detected cars."""
        output_video_path = f"{self.output_path}/inference_simple.mp4"
        print(f"Rendering simple video to: {output_video_path}")
        
        with sv.VideoSink(target_path=output_video_path, video_info=self.video_info) as sink:
            frames_generator = sv.get_video_frames_generator(source_path=self.local_video_path)
            for frame_no, frame in enumerate(tqdm(frames_generator, total=self.video_info.total_frames, desc="Rendering Simple Video")):
                annotated_frame = frame.copy()
                if frame_no in self.car_detections:
                    car_detections = self.car_detections[frame_no]
                    car_id = self.get_ids(car_detections)
                    
                    if car_id in self.analysis:
                        label = self.add_label(frame_no)
                        color = self.get_color(label)
                        
                        if color == "green":
                            annotated_frame = self.label_annotator_green.annotate(annotated_frame, car_detections, labels=label)
                            annotated_frame = self.box_annotator_green.annotate(annotated_frame, car_detections)
                        elif color == "red":
                            annotated_frame = self.label_annotator_red.annotate(annotated_frame, car_detections, labels=label)
                            annotated_frame = self.box_annotator_red.annotate(annotated_frame, car_detections)
                        else: # gray
                            annotated_frame = self.label_annotator_gray.annotate(annotated_frame, car_detections, labels=label)
                            annotated_frame = self.box_annotator_gray.annotate(annotated_frame, car_detections)

                sink.write_frame(annotated_frame)
        print("Simple video rendering complete.")

    def render_video_advanced(self):
        """Renders a detailed video with all zones, detections, and tracking info."""
        output_video_path = f"{self.output_path}/inference_advanced.mp4"
        print(f"Rendering advanced video to: {output_video_path}")

        with sv.VideoSink(target_path=output_video_path, video_info=self.video_info) as sink:
            frames_generator = sv.get_video_frames_generator(source_path=self.local_video_path)
            for frame_no, frame in enumerate(tqdm(frames_generator, total=self.video_info.total_frames, desc="Rendering Advanced Video")):
                annotated_frame = frame.copy()
                annotated_frame = self.stopzone_annotator.annotate(scene=annotated_frame)
                annotated_frame = self.outzone_annotator.annotate(scene=annotated_frame)
                annotated_frame = self.road_annotator.annotate(scene=annotated_frame)

                point_x, point_y = self.point
                cv2.circle(annotated_frame, (int(point_x), int(point_y)), 10, (0, 255, 255), -1)

                if frame_no in self.car_detections:
                    car_detections = self.car_detections[frame_no]
                    car_id = self.get_ids(car_detections)
                    
                    if car_id in self.analysis:
                        car_center = self.get_car_center(frame_no)
                        if car_center:
                            cv2.circle(annotated_frame, (int(car_center[0]), int(car_center[1])), 10, (0, 255, 0), -1)
                            cv2.line(annotated_frame, (int(point_x), int(point_y)), (int(car_center[0]), int(car_center[1])), (0, 255, 0), 2)
                        
                        label = self.add_label(frame_no)
                        color = self.get_color(label)
                        
                        if color == "green":
                            annotated_frame = self.label_annotator_green.annotate(annotated_frame, car_detections, labels=label)
                            annotated_frame = self.box_annotator_green.annotate(annotated_frame, car_detections)
                        elif color == "red":
                            annotated_frame = self.label_annotator_red.annotate(annotated_frame, car_detections, labels=label)
                            annotated_frame = self.box_annotator_red.annotate(annotated_frame, car_detections)
                        else:
                            annotated_frame = self.label_annotator_gray.annotate(annotated_frame, car_detections, labels=label)
                            annotated_frame = self.box_annotator_gray.annotate(annotated_frame, car_detections)

                if frame_no in self.wheel_detections and 'car_id' in locals() and car_id in self.reported_wheels:
                    wheel_detections = self.wheel_detections[frame_no]
                    # Only show the front-most wheel that triggered the analysis
                    wheel_detections = self.filter_detections(wheel_detections, self.reported_wheels[car_id])
                    annotated_frame = self.box_annotator_white.annotate(annotated_frame, wheel_detections)
                
                sink.write_frame(annotated_frame)
        print("Advanced video rendering complete.")

    def filter_detections(self, detections, tracker_id):
        """Filters detections to only include a specific tracker ID."""
        if detections is None or len(detections) == 0:
            return detections
        return detections[detections.tracker_id == tracker_id]

    def get_color(self, label):
        """Returns a color based on the label content."""
        if not label: return "gray"
        if "Stopped" in label[0]: return "green"
        if "Failed" in label[0]: return "red"
        return "gray"

    def add_label(self, frame_no):
        """Constructs a display label for a given car in a frame."""
        car_id = self.get_ids(self.car_detections[frame_no])
        if car_id in self.analysis:
            status = self.analysis[car_id]['Status']
            if car_id in self.car_left_zone and frame_no >= self.car_left_zone[car_id]:
                return [f"{status} (id:{car_id})"]
            else:
                return [f"Tracking (id:{car_id})"]
        # Fallback for when car_id is not in analysis yet
        elif car_id is not None:
             return [f"Tracking (id:{car_id})"]
        return []


if __name__ == '__main__':
    # --- Configuration ---
    VIDEO_NAME = "stop_sign_video"
    # Make sure this path points to your video file
    LOCAL_VIDEO_PATH = "video2.mp4" 
    # **IMPORTANT**: Download the 'yolo11n.pt' file from the GitHub repo's releases
    # and place it in your project folder or provide the full path here.
    # https://github.com/andBabaev/wheel-detector/releases
    WHEEL_MODEL_PATH = "yolo11n.pt" 

    # --- Execution ---
    if not os.path.exists(LOCAL_VIDEO_PATH):
        print(f"ERROR: Video file not found at '{LOCAL_VIDEO_PATH}'")
        print("Please update the LOCAL_VIDEO_PATH variable.")
    elif not os.path.exists(WHEEL_MODEL_PATH):
        print(f"ERROR: Wheel model not found at '{WHEEL_MODEL_PATH}'")
        print("Please download 'yolo11n.pt' from the GitHub repository and update the WHEEL_MODEL_PATH variable.")
    else:
        # 1. Initialize the video processor
        stop_video_processor = StopVideo(
            video_name=VIDEO_NAME,
            local_video_path=LOCAL_VIDEO_PATH,
            wheel_model_path=WHEEL_MODEL_PATH
        )

        # 2. Run inference to detect objects in every frame
        stop_video_processor.inference()

        # 3. Analyze the results to determine stop status
        stop_video_processor.analyze()

        # Print the final analysis results
        print("\n--- Final Analysis ---")
        for car_id, data in stop_video_processor.analysis.items():
            print(f"Car ID: {car_id}, Status: {data['Status']}")
        print("----------------------\n")

        # 4. Render output videos
        # stop_video_processor.render_video_simple()
        stop_video_processor.render_video_advanced()

        print("\nProcessing finished successfully!")
