import collections
import logging
from utils.helpers import handle_error
from common.constants import (
    EAR_THRESHOLD, EAR_CONSEC_FRAMES, REQUIRED_BLINKS,
    LIVENESS_TIMEOUT_FRAMES, HEAD_MOVEMENT_FRAMES,
    POSE_HISTORY_FRAMES, LOOK_LEFT_RIGHT_ANGLE_THRESH
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

    def start_checking(self):
        """Start a new liveness check."""
        self.reset()
        self.is_checking = True
        logging.info("Starting new liveness check")

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
            return False, "Not checking"

        if self.liveness_check_frame_counter > LIVENESS_TIMEOUT_FRAMES:
            self.is_checking = False
            return False, "Timeout"

        if self.head_movement_ok is False:
            return False, "No head movement"

        if self.head_pose_variation_ok is False:
            return False, "No pose variation"

        blinks_ok = self.blinks_detected_count >= REQUIRED_BLINKS
        head_move_ok = self.head_movement_ok is True
        look_lr_ok = self.looked_left and self.looked_right

        if blinks_ok and head_move_ok and look_lr_ok:
            self.liveness_passed = True
            self.is_checking = False
            return True, "Passed"

        status_parts = [
            f"B:{self.blinks_detected_count}/{REQUIRED_BLINKS}",
            f"HM:{'?' if self.head_movement_ok is None else ('OK' if head_move_ok else 'NO')}",
            f"L/R:{'OK' if look_lr_ok else ('L' if self.looked_left else ('R' if self.looked_right else 'N'))}",
            f"{self.liveness_check_frame_counter}/{LIVENESS_TIMEOUT_FRAMES}F"
        ]
        return False, f"Checking ({', '.join(status_parts)})"

    def increment_frame_counter(self):
        """Increment the frame counter for timeout checking."""
        if self.is_checking:
            self.liveness_check_frame_counter += 1 