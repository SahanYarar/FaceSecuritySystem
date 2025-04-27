import logging
import numpy as np
from scipy.spatial.distance import euclidean
import math
import traceback
import cv2

def shape_to_np(shape, dtype="int"):
    """
    Convert a dlib shape object to a numpy array.
    
    Args:
        shape: dlib shape object containing facial landmarks
        dtype: data type for the output array (default: int)
    
    Returns:
        numpy.ndarray: Array of (x, y) coordinates
    """
    if shape is None:
        return np.zeros((0, 2), dtype=dtype)
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def handle_error(error_msg, return_value=None, log_level=logging.ERROR):
    """
    Common error handling function that logs the error and returns a specified value.
    
    Args:
        error_msg (str): Error message to log
        return_value: Value to return after logging the error
        log_level (int): Logging level to use (default: logging.ERROR)
    
    Returns:
        The specified return_value
    """
    logging.log(log_level, f"{error_msg}\n{traceback.format_exc()}")
    return return_value

def eye_aspect_ratio(eye_landmarks):
    """
    Calculate the eye aspect ratio given eye landmarks.
    
    Args:
        eye_landmarks: Array of 6 (x, y) coordinates of facial landmarks for the eye
    
    Returns:
        float: The eye aspect ratio
    """
    try:
        # Compute the euclidean distances between the vertical eye landmarks
        A = euclidean(eye_landmarks[1], eye_landmarks[5])
        B = euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception as e:
        return handle_error(f"Error calculating eye aspect ratio: {str(e)}", 0.0)

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First bounding box (dlib.rectangle or list/tuple of coordinates)
        box2: Second bounding box (dlib.rectangle or list/tuple of coordinates)
    
    Returns:
        float: IoU value
    """
    try:
        # Convert dlib rectangles to coordinate arrays if needed
        if hasattr(box1, 'left'):  # Check if it's a dlib rectangle
            box1_coords = [box1.left(), box1.top(), box1.right(), box1.bottom()]
        else:
            box1_coords = box1

        if hasattr(box2, 'left'):  # Check if it's a dlib rectangle
            box2_coords = [box2.left(), box2.top(), box2.right(), box2.bottom()]
        else:
            box2_coords = box2

        # Determine the coordinates of the intersection rectangle
        x_left = max(box1_coords[0], box2_coords[0])
        y_top = max(box1_coords[1], box2_coords[1])
        x_right = min(box1_coords[2], box2_coords[2])
        y_bottom = min(box1_coords[3], box2_coords[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate area of both bounding boxes
        box1_area = (box1_coords[2] - box1_coords[0]) * (box1_coords[3] - box1_coords[1])
        box2_area = (box2_coords[2] - box2_coords[0]) * (box2_coords[3] - box2_coords[1])

        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return max(0.0, min(iou, 1.0))
    except Exception as e:
        return handle_error(f"Error calculating IoU: {str(e)}", 0.0)

def calculate_centroid(landmarks):
    """
    Calculate the centroid of facial landmarks.
    
    Args:
        landmarks: Array of (x, y) coordinates of facial landmarks
    
    Returns:
        tuple: (x, y) coordinates of the centroid
    """
    try:
        x_coords = [p[0] for p in landmarks]
        y_coords = [p[1] for p in landmarks]
        centroid_x = sum(x_coords) / len(landmarks)
        centroid_y = sum(y_coords) / len(landmarks)
        return (centroid_x, centroid_y)
    except Exception as e:
        return handle_error(f"Error calculating centroid: {str(e)}", (0, 0))

def calculate_head_pose(landmarks):
    """
    Calculate approximate head pose angles from facial landmarks.
    
    Args:
        landmarks: Array of (x, y) coordinates of facial landmarks
    
    Returns:
        tuple: (pitch, yaw, roll) angles in degrees
    """
    try:
        # Calculate nose tip to center distance
        nose_tip = landmarks[30]
        nose_bridge = landmarks[27]
        nose_length = euclidean(nose_tip, nose_bridge)
        
        # Calculate left-right eye centers
        left_eye = np.mean([landmarks[36], landmarks[39]], axis=0)
        right_eye = np.mean([landmarks[42], landmarks[45]], axis=0)
        eye_distance = euclidean(left_eye, right_eye)
        
        # Estimate yaw from nose deviation
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        yaw = math.degrees(math.atan2(nose_tip[0] - face_center_x, nose_length))
        
        # Estimate pitch from vertical nose position
        face_center_y = (left_eye[1] + right_eye[1]) / 2
        pitch = math.degrees(math.atan2(nose_tip[1] - face_center_y, nose_length))
        
        # Estimate roll from eye angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        roll = math.degrees(math.atan2(dy, dx))
        
        return (pitch, yaw, roll)
    except Exception as e:
        return handle_error(f"Error calculating head pose: {str(e)}", (0, 0, 0))

def resize_frame(frame, target_width):
    """
    Resize a frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        target_width: Desired width
    
    Returns:
        numpy.ndarray: Resized frame
    """
    try:
        height, width = frame.shape[:2]
        ratio = target_width / float(width)
        target_height = int(height * ratio)
        return cv2.resize(frame, (target_width, target_height))
    except Exception as e:
        return handle_error(f"Error resizing frame: {str(e)}", frame) 