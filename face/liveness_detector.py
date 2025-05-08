import collections
import logging
import time
from utils.helpers import handle_error
from common.constants import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES, REQUIRED_BLINKS,
    LIVENESS_TIMEOUT_FRAMES, HEAD_MOVEMENT_FRAMES,
    POSE_HISTORY_FRAMES, LOOK_LEFT_RIGHT_ANGLE_THRESH,
    COLOR_WHITE, COLOR_RED, COLOR_GREEN, COLOR_YELLOW,
    POSE_CENTER_THRESHOLD, LIVENESS_DURATION_SECONDS,
    REQUIRED_SCORE
)

class LivenessDetector:
    """
    Liveness detection system that verifies a real person is present.
    
    Components:
    1. Eye Blink Detection:
       - Uses Eye Aspect Ratio (EAR) = (p2-p6 + p3-p5) / (2 * (p1-p4))
       - p1-p6 are eye landmarks (left to right)
       - EAR < threshold indicates eye closure
       - Requires multiple valid blinks
    
    2. Head Movement Detection:
       - Centroid = average of all facial landmarks
       - Tracks movement using Euclidean distance
       - Requires natural head movement
       - Prevents static photo spoofing
    
    3. Head Pose Variation:
       - Yaw: Rotation around vertical axis (left/right)
       - Pitch: Rotation around horizontal axis (up/down)
       - Roll: Rotation around depth axis (tilt)
       - Uses 3D facial landmarks for calculation
       - Requires left/right head turns
    
    4. Scoring System:
       - Blink: +2 points per valid blink
       - Head movement: +1 point per valid movement
       - Pose variation: +1 point per valid turn
       - Requires minimum score for verification
    """
    def __init__(self):
        # Liveness state variables
        self.liveness_passed = False  # Overall liveness status
        self.ear_consec_counter = 0  # Consecutive frames with low EAR
        self.liveness_check_frame_counter = 0  # Frame counter for timeout
        self.blinks_detected_count = 0  # Number of valid blinks detected
        self.blink_score = 0  # Score from blink detection
        
        # Head movement tracking
        # Centroid = (avg_x, avg_y) of all facial landmarks
        # Movement = Euclidean distance between consecutive centroids
        self.centroid_history = collections.deque(maxlen=HEAD_MOVEMENT_FRAMES)  # Recent face positions
        self.head_movement_ok = None  # Head movement verification status
        
        # Head pose tracking
        # Pose angles calculated using:
        # 1. Nose tip as reference point
        # 2. Eye corners for horizontal alignment
        # 3. Mouth corners for vertical alignment
        self.pose_history = collections.deque(maxlen=POSE_HISTORY_FRAMES)  # Recent head poses
        self.head_pose_variation_ok = None  # Pose variation verification status
        self.initial_yaw = None  # Initial head yaw angle
        self.looked_left = False  # Left look verification
        self.looked_right = False  # Right look verification
        
        # System state
        self.is_checking = False  # Whether liveness check is active
        self.liveness_passed_time = None  # Timestamp of successful verification
        self.stable_match_name = None  # Name of person being verified
        
        # UI status
        self.system_status = {
            "liveness": "Waiting for Face Recognition",
            "liveness_color": COLOR_WHITE
        }

    def reset(self):
        """
        Reset all liveness detection state.
        
        Called when:
        1. Starting new liveness check
        2. Liveness check fails
        3. Liveness check times out
        4. System mode changes
        """
        self.liveness_passed = False
        self.ear_consec_counter = 0
        self.liveness_check_frame_counter = 0
        self.blinks_detected_count = 0
        self.centroid_history.clear()
        self.head_movement_ok = None
        self.pose_history.clear()
        self.head_pose_variation_ok = None
        self.initial_yaw = None
        self.looked_left = False
        self.looked_right = False
        self.is_checking = False
        self.blink_score = 0
        self.liveness_passed_time = None
        self.system_status = {
            "liveness": "Waiting for Face Recognition",
            "liveness_color": COLOR_WHITE
        }

    def start_checking(self):
        """
        Start a new liveness check.
        
        Process:
        1. Reset all state variables
        2. Set checking flag
        3. Update UI status
        
        Side Effects:
            - Resets all detection counters
            - Updates system status
            - Changes UI color to yellow
        """
        self.reset()
        self.is_checking = True
        logging.info("Starting new liveness check")
        self.system_status["liveness"] = "Check Started"
        self.system_status["liveness_color"] = COLOR_YELLOW

    def set_stable_match_name(self, name):
        """
        Set the name of the person being verified.
        
        Args:
            name: Name of the recognized person
            
        Side Effects:
            - Updates system status with name
            - Changes UI color to yellow
        """
        self.stable_match_name = name
        if self.is_checking:
            self.system_status["liveness"] = f"{name}: Checking"
            self.system_status["liveness_color"] = COLOR_YELLOW

    def update_ear(self, ear_value, blink_data=None):
        """
        Update eye aspect ratio based liveness detection.
        
        Eye Aspect Ratio (EAR) Calculation:
        1. For each eye, get 6 landmarks:
           p1, p2, p3, p4, p5, p6 (left to right)
        2. Calculate vertical distances:
           v1 = distance(p2, p6)
           v2 = distance(p3, p5)
        3. Calculate horizontal distance:
           h = distance(p1, p4)
        4. EAR = (v1 + v2) / (2 * h)
        
        Blink Detection:
        1. EAR < threshold indicates eye closure
        2. Consecutive frames with low EAR = blink
        3. Minimum frames required for valid blink
        4. Maximum frames allowed for single blink
        
        Args:
            ear_value: Current eye aspect ratio
            blink_data: Optional blink tracking data
            
        Side Effects:
            - Updates blink counters
            - Updates blink score
            - Logs blink detection
        """
        if not self.is_checking:
            return

        if ear_value is not None:
            if ear_value < EAR_THRESHOLD:
                self.ear_consec_counter += 1
            else:
                if self.ear_consec_counter >= EAR_CONSEC_FRAMES:
                    self.blinks_detected_count += 1
                    self.blink_score += 2  # Award points for valid blink
                    logging.info(f"Blink detected! Total: {self.blinks_detected_count}/{REQUIRED_BLINKS}, Score: {self.blink_score}")
                self.ear_consec_counter = 0

            # Penalize for long eye closure
            if self.ear_consec_counter > EAR_CONSEC_FRAMES * 2:
                self.blink_score = max(0, self.blink_score - 1)  # Penalize for long eye closure

    def update_centroid(self, centroid, check_result):
        """
        Update head movement based liveness detection.
        
        Centroid Calculation:
        1. Get all facial landmarks (68 points)
        2. Calculate average x and y coordinates:
           centroid_x = sum(x_i) / n
           centroid_y = sum(y_i) / n
        
        Movement Detection:
        1. Track centroid position over time
        2. Calculate Euclidean distance between consecutive positions
        3. Movement = sqrt((x2-x1)² + (y2-y1)²)
        4. Check if movement exceeds threshold
        
        Args:
            centroid: Current face centroid coordinates (x, y)
            check_result: Result of head movement check
            
        Side Effects:
            - Updates centroid history
            - Updates movement status
            - Logs movement detection
        """
        if not self.is_checking:
            return

        if centroid is not None:
            self.centroid_history.append(centroid)
            if check_result is not None and self.head_movement_ok is None:
                self.head_movement_ok = check_result
                if check_result:
                    logging.info("Head movement detected")

    def update_pose(self, pose_angles, pose_check_result, current_yaw):
        """
        Update head pose based liveness detection.
        
        Pose Calculation:
        1. Yaw (left/right rotation):
           - Use nose tip and eye corners
           - Calculate angle between nose and eye line
           - Positive = right turn, Negative = left turn
        
        2. Pitch (up/down rotation):
           - Use nose tip and mouth corners
           - Calculate angle between nose and mouth line
           - Positive = up, Negative = down
        
        3. Roll (tilt):
           - Use eye corners and mouth corners
           - Calculate angle between eye line and mouth line
           - Positive = clockwise, Negative = counter-clockwise
        
        Pose Variation Check:
        1. Track pose angles over time
        2. Check for significant changes
        3. Verify left/right head turns
        4. Ensure return to center position
        
        Args:
            pose_angles: Current head pose angles (pitch, yaw, roll)
            pose_check_result: Result of pose variation check
            current_yaw: Current yaw angle
            
        Side Effects:
            - Updates pose history
            - Updates look direction status
            - Logs pose changes
        """
        if not self.is_checking:
            return

        if pose_angles is not None:
            self.pose_history.append(pose_angles)
            if pose_check_result is not None and self.head_pose_variation_ok is None:
                self.head_pose_variation_ok = pose_check_result
                if pose_check_result:
                    logging.info("Head pose variation detected")

            if self.initial_yaw is None:
                self.initial_yaw = current_yaw
                logging.info(f"Liveness: Initial Yaw angle set to {self.initial_yaw:.1f} degrees")

            if self.initial_yaw is not None and current_yaw is not None:
                # Calculate yaw difference from initial position
                yaw_diff = current_yaw - self.initial_yaw
                
                logging.debug(f"Yaw: {current_yaw:.1f}, Initial: {self.initial_yaw:.1f}, Diff: {yaw_diff:.1f}, Left: {self.looked_left}, Right: {self.looked_right}")
                
                if not self.looked_left and yaw_diff > LOOK_LEFT_RIGHT_ANGLE_THRESH:
                    self.looked_left = True
                    logging.info(f"Liveness: Looked left detected (Yaw: {current_yaw:.1f}, Diff: {yaw_diff:.1f})")
                
                if not self.looked_right and yaw_diff < -LOOK_LEFT_RIGHT_ANGLE_THRESH:
                    self.looked_right = True
                    logging.info(f"Liveness: Looked right detected (Yaw: {current_yaw:.1f}, Diff: {yaw_diff:.1f})")

                # Reset if head returns to center position
                if abs(yaw_diff) < POSE_CENTER_THRESHOLD:
                    if self.looked_left and not self.looked_right:
                        logging.info("Liveness: Head returned to center after left look")
                    elif self.looked_right and not self.looked_left:
                        logging.info("Liveness: Head returned to center after right look")
                    # Reset initial yaw when returning to center to make next movement easier to detect
                    self.initial_yaw = current_yaw
                    logging.info(f"Liveness: Reset initial yaw to {self.initial_yaw:.1f} degrees")

    def check_liveness(self):
        """
        Check if all liveness requirements are met.
        
        Process:
        1. Check if liveness check is active
        2. Check for timeout
        3. Verify all requirements:
           - Sufficient blinks
           - Head movement
           - Left/right looks
           - Minimum score
        4. Update status and return result
        
        Returns:
            tuple: (passed, status_data)
            - passed: Boolean indicating liveness passed
            - status_data: Dictionary with detailed status information
        """
        if not self.is_checking:
            return False, {
                "status": "not_checking",
                "name": self.stable_match_name,
                "blinks": self.blinks_detected_count,
                "required_blinks": REQUIRED_BLINKS,
                "head_movement": self.head_movement_ok,
                "looked_left": self.looked_left,
                "looked_right": self.looked_right,
                "frames_remaining": LIVENESS_TIMEOUT_FRAMES - self.liveness_check_frame_counter,
                "current_yaw": self.initial_yaw if self.initial_yaw is not None else None,
                "score": self.blink_score,
                "required_score": REQUIRED_SCORE
            }

        # Check if liveness has expired
        if self.liveness_passed and self.liveness_passed_time is not None:
            elapsed_time = time.time() - self.liveness_passed_time
            remaining_time = max(0, LIVENESS_DURATION_SECONDS - elapsed_time)
            
            if elapsed_time > LIVENESS_DURATION_SECONDS:
                self.reset()
                return False, {
                    "status": "expired",
                    "name": self.stable_match_name,
                    "elapsed_time": elapsed_time,
                    "score": self.blink_score,
                    "required_score": REQUIRED_SCORE
                }
            else:
                # Return passed status with remaining time
                return True, {
                    "status": "passed",
                    "name": self.stable_match_name,
                    "expires_in": remaining_time,
                    "blinks": self.blinks_detected_count,
                    "required_blinks": REQUIRED_BLINKS,
                    "head_movement": self.head_movement_ok,
                    "looked_left": self.looked_left,
                    "looked_right": self.looked_right,
                    "should_open_door": True,
                    "score": self.blink_score,
                    "required_score": REQUIRED_SCORE
                }

        # Timeout check
        if self.liveness_check_frame_counter > LIVENESS_TIMEOUT_FRAMES:
            self.is_checking = False
            return False, {
                "status": "timeout",
                "name": self.stable_match_name,
                "score": self.blink_score,
                "required_score": REQUIRED_SCORE
            }

        # --- Decision Making Phase ---
        # Get current states
        blinks_ok = self.blinks_detected_count >= REQUIRED_BLINKS
        head_move_ok = self.head_movement_ok is True
        look_lr_ok = self.looked_left and self.looked_right
        score_ok = self.blink_score >= REQUIRED_SCORE

        # --- Failure Conditions (Priority) ---
        # 1. Timeout (checked above)

        # 2. Definite centroid immobility detected
        if self.head_movement_ok is False:
            self.reset()
            return False, {
                "status": "insufficient_head_movement",
                "name": self.stable_match_name,
                "score": self.blink_score,
                "required_score": REQUIRED_SCORE
            }

        # 3. Definite pose immobility detected (photo suspicion)
        if self.head_pose_variation_ok is False:
            self.reset()
            return False, {
                "status": "low_pose_variation",
                "name": self.stable_match_name,
                "score": self.blink_score,
                "required_score": REQUIRED_SCORE
            }

        # --- Success Condition ---
        if blinks_ok and head_move_ok and look_lr_ok and score_ok:
            self.liveness_passed = True
            self.liveness_passed_time = time.time()  # Record the time when liveness was passed
            return True, {
                "status": "passed",
                "name": self.stable_match_name,
                "expires_in": LIVENESS_DURATION_SECONDS,
                "blinks": self.blinks_detected_count,
                "required_blinks": REQUIRED_BLINKS,
                "head_movement": self.head_movement_ok,
                "looked_left": self.looked_left,
                "looked_right": self.looked_right,
                "should_open_door": True,
                "score": self.blink_score,
                "required_score": REQUIRED_SCORE
            }

        # --- In Progress Status ---
        self.liveness_check_frame_counter += 1
        return False, {
            "status": "in_progress",
            "name": self.stable_match_name,
            "blinks": self.blinks_detected_count,
            "required_blinks": REQUIRED_BLINKS,
            "head_movement": self.head_movement_ok,
            "looked_left": self.looked_left,
            "looked_right": self.looked_right,
            "frames_remaining": LIVENESS_TIMEOUT_FRAMES - self.liveness_check_frame_counter,
            "score": self.blink_score,
            "required_score": REQUIRED_SCORE
        }

    def increment_frame_counter(self):
        """
        Increment the frame counter for timeout checking.
        
        Side Effects:
            - Updates frame counter
            - May trigger timeout check
        """
        if self.is_checking:
            self.liveness_check_frame_counter += 1 