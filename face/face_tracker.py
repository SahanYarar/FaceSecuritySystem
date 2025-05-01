import logging
import time
from utils.helpers import calculate_iou
from common.constants import STREAK_THRESHOLD, RECOG_DIST_THRESH

class FaceTracker:
    """
    Face tracking system that maintains state about detected faces across frames.
    
    Key Components:
    1. Face Position Tracking: Using IoU (Intersection over Union)
    2. Recognition State: Tracks recognition confidence and streak
    3. Face Descriptor Storage: Maintains last known face descriptor
    4. Timeout Handling: Resets state after inactivity
    """
    def __init__(self):
        # Recognition state variables
        self.candidate_name = None  # Current best match name
        self.recognition_streak_count = 0  # Consecutive frames with same match
        self.stable_match_name = None  # Confirmed identity after sufficient streak
        
        # Face tracking variables
        self.last_known_rect = None  # Last known face position (dlib.rectangle)
        self.last_known_descriptor = None  # Last computed face descriptor
        self.last_processed_landmarks = None  # Last detected facial landmarks
        
        # Frame processing variables
        self.scale_factor = 1.0  # Scale between original and processed frame
        self.last_detection_time = 0  # Timestamp of last face detection
        self.current_process_frame_size = (0, 0)  # Size of processed frame (H, W)
        
        # Face loss tracking
        self.no_face_frames = 0  # Consecutive frames without face detection
        self.max_no_face_frames = 5  # Frames to wait before resetting state

    def reset(self):
        """
        Reset all tracking state variables.
        Called when:
        1. Face tracking is lost
        2. System timeout occurs
        3. Mode changes
        """
        self.candidate_name = None
        self.recognition_streak_count = 0
        self.stable_match_name = None
        self.last_known_rect = None
        self.last_known_descriptor = None
        self.last_processed_landmarks = None
        self.no_face_frames = 0

    def update_recognition(self, face_descriptor, known_faces, face_processor):
        """
        Update face recognition state using percentage-based verification.
        
        Process:
        1. Compare current face with all known faces
        2. Find best match based on similarity percentage
        3. Update recognition streak if match is consistent
        4. Confirm identity if streak threshold is reached
        
        Args:
            face_descriptor: Current face's 128D descriptor
            known_faces: Dictionary of {name: descriptor} pairs
            face_processor: FaceProcessor instance for comparison
            
        Returns:
            bool: True if stable match is confirmed
        """
        if face_descriptor is None or not known_faces:
            # Only reset if we've had too many consecutive frames without a face
            if self.no_face_frames >= self.max_no_face_frames:
                self.candidate_name = None
                self.recognition_streak_count = 0
                self.stable_match_name = None
                self.last_known_descriptor = None
            return False

        # Reset no face counter since we found a face
        self.no_face_frames = 0
        self.last_known_descriptor = face_descriptor

        # Find best match using percentage
        best_match_name = None
        best_match_percentage = 0.0

        for name, known_descriptor in known_faces.items():
            percentage = face_processor.compare_faces(face_descriptor, known_descriptor)
            if percentage > best_match_percentage:
                best_match_percentage = percentage
                best_match_name = name

        # Check if match percentage is good enough (e.g., > 45%)
        if best_match_percentage > 50:
            if self.candidate_name == best_match_name:
                self.recognition_streak_count += 1
            else:
                # Only change candidate if we've had enough consecutive frames without a match
                if self.recognition_streak_count < 3:  # Allow some frames to switch candidates
                    self.candidate_name = best_match_name
                    self.recognition_streak_count = 1

            # If we have enough consecutive matches, consider it stable
            if self.recognition_streak_count >= STREAK_THRESHOLD:
                self.stable_match_name = self.candidate_name
                return True
        else:
            # Only reset if we've had enough consecutive frames without a good match
            if self.recognition_streak_count > 0:
                self.recognition_streak_count -= 1
            else:
                self.candidate_name = None
                self.stable_match_name = None

        return False

    def update_tracking(self, current_rect):
        """
        Update face tracking state using IoU (Intersection over Union).
        
        Process:
        1. Calculate IoU between current and last known face position
        2. Update tracking if IoU is above threshold
        3. Reset tracking if IoU is too low
        
        Args:
            current_rect: Current face position (dlib.rectangle)
            
        Returns:
            bool: True if tracking is maintained
        """
        if current_rect is None:
            self.no_face_frames += 1
            return self.no_face_frames < self.max_no_face_frames

        # Reset no face counter since we found a face
        self.no_face_frames = 0

        # If this is the first face detected, initialize tracking
        if self.last_known_rect is None:
            self.last_known_rect = current_rect
            return True

        # Calculate IoU between current and last known face
        iou = calculate_iou(current_rect, self.last_known_rect)
        
        # If IoU is too low, face tracking is lost
        if iou < 0.2:  # Lower threshold for more stability
            logging.info(f"Face tracking lost - IoU too low: {iou}")
            return False

        # Update last known face position
        self.last_known_rect = current_rect
        return True

    def check_timeout(self):
        """
        Check if face detection has timed out.
        
        Process:
        1. Calculate time since last detection
        2. Reset state if timeout threshold is exceeded
        3. Update last detection time
        
        Returns:
            bool: True if timeout occurred
        """
        time_since_last_detection = time.time() - self.last_detection_time
        if (self.stable_match_name or self.candidate_name or self.last_known_rect) and time_since_last_detection > 2.0:
            self.reset()
            return True
        return False 