import collections
import logging
from utils.helpers import handle_error
from common.constants import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES, REQUIRED_BLINKS,
    LIVENESS_TIMEOUT_FRAMES, HEAD_MOVEMENT_FRAMES,
    POSE_HISTORY_FRAMES, LOOK_LEFT_RIGHT_ANGLE_THRESH,
    COLOR_WHITE, COLOR_RED, COLOR_GREEN, COLOR_YELLOW
)

class LivenessDetector:
    def __init__(self):
        self.liveness_passed = False
        self.ear_consec_counter = 0
        self.liveness_check_frame_counter = 0
        self.blinks_detected_count = 0
        self.centroid_history = collections.deque(maxlen=HEAD_MOVEMENT_FRAMES)
        self.head_movement_ok = None
        self.pose_history = collections.deque(maxlen=POSE_HISTORY_FRAMES)
        self.head_pose_variation_ok = None
        self.initial_yaw = None
        self.looked_left = False
        self.looked_right = False
        self.is_checking = False
        self.system_status = {
            "liveness": "Waiting for Face Recognition",
            "liveness_color": COLOR_WHITE
        }
        self.stable_match_name = None

    def reset(self):
        """Reset all liveness detection state."""
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
        self.system_status = {
            "liveness": "Waiting for Face Recognition",
            "liveness_color": COLOR_WHITE
        }

    def start_checking(self):
        """Start a new liveness check."""
        self.reset()
        self.is_checking = True
        logging.info("Starting new liveness check")
        self.system_status["liveness"] = "Check Started"
        self.system_status["liveness_color"] = COLOR_YELLOW

    def set_stable_match_name(self, name):
        """Set the name of the person being verified."""
        self.stable_match_name = name
        if self.is_checking:
            self.system_status["liveness"] = f"{name}: Checking"
            self.system_status["liveness_color"] = COLOR_YELLOW

    def update_ear(self, ear_value):
        """Update eye aspect ratio based liveness detection."""
        if not self.is_checking:
            return

        if ear_value is not None:
            if ear_value < EAR_THRESHOLD:
                self.ear_consec_counter += 1
            else:
                if self.ear_consec_counter >= EAR_CONSEC_FRAMES:
                    self.blinks_detected_count += 1
                    logging.info(f"Blink detected! Total: {self.blinks_detected_count}/{REQUIRED_BLINKS}")
                self.ear_consec_counter = 0

    def update_centroid(self, centroid, check_result):
        """Update head movement based liveness detection."""
        if not self.is_checking:
            return

        if centroid is not None:
            self.centroid_history.append(centroid)
            if check_result is not None and self.head_movement_ok is None:
                self.head_movement_ok = check_result
                if check_result:
                    logging.info("Head movement detected")

    def update_pose(self, pose_angles, pose_check_result, current_yaw):
        """Update head pose based liveness detection."""
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
                if not self.looked_left and current_yaw < self.initial_yaw - LOOK_LEFT_RIGHT_ANGLE_THRESH:
                    self.looked_left = True
                    logging.info(f"Liveness: Looked left detected (Yaw: {current_yaw:.1f})")
                if not self.looked_right and current_yaw > self.initial_yaw + LOOK_LEFT_RIGHT_ANGLE_THRESH:
                    self.looked_right = True
                    logging.info(f"Liveness: Looked right detected (Yaw: {current_yaw:.1f})")

    def check_liveness(self):
        """Check if all liveness requirements are met."""
        if not self.is_checking:
            return False, {
                "status": "not_checking",
                "name": self.stable_match_name,
                "blinks": self.blinks_detected_count,
                "required_blinks": REQUIRED_BLINKS,
                "head_movement": self.head_movement_ok,
                "looked_left": self.looked_left,
                "looked_right": self.looked_right,
                "frames_remaining": LIVENESS_TIMEOUT_FRAMES - self.liveness_check_frame_counter
            }

        # Timeout check
        if self.liveness_check_frame_counter > LIVENESS_TIMEOUT_FRAMES:
            self.is_checking = False
            return False, {
                "status": "timeout",
                "name": self.stable_match_name
            }

        # --- Decision Making Phase ---
        # Get current states
        blinks_ok = self.blinks_detected_count >= REQUIRED_BLINKS
        head_move_ok = self.head_movement_ok is True
        look_lr_ok = self.looked_left and self.looked_right

        # --- Failure Conditions (Priority) ---
        # 1. Timeout (checked above)

        # 2. Definite centroid immobility detected
        if self.head_movement_ok is False:
            self.reset()
            return False, {
                "status": "insufficient_head_movement",
                "name": self.stable_match_name
            }

        # 3. Definite pose immobility detected (photo suspicion)
        if self.head_pose_variation_ok is False:
            self.reset()
            return False, {
                "status": "low_pose_variation",
                "name": self.stable_match_name
            }

        # --- Success Condition ---
        if blinks_ok and head_move_ok and look_lr_ok:
            self.liveness_passed = True
            return True, {
                "status": "passed",
                "name": self.stable_match_name
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
            "frames_remaining": LIVENESS_TIMEOUT_FRAMES - self.liveness_check_frame_counter
        }

    def increment_frame_counter(self):
        """Increment the frame counter for timeout checking."""
        if self.is_checking:
            self.liveness_check_frame_counter += 1 