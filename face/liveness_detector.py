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

        # Timeout check
        if self.liveness_check_frame_counter > LIVENESS_TIMEOUT_FRAMES:
            self.is_checking = False
            return False, "Timeout"

        # Check individual conditions
        blinks_ok = self.blinks_detected_count >= REQUIRED_BLINKS
        head_move_ok = self.head_movement_ok is True  # Must be True, not just not None
        look_lr_ok = self.looked_left and self.looked_right

        # --- Failure Conditions (Priority) ---
        # 1. Timeout (checked above)

        # 2. Definite centroid immobility detected
        if self.head_movement_ok is False:
            logging.warning("Canlılık BAŞARISIZ: Kafa merkezi hareketi yetersiz!")
            return False, "Başarısız (Hareketsiz?)"

        # 3. Definite pose immobility detected (photo suspicion)
        if self.head_pose_variation_ok is False:
            logging.warning("Canlılık BAŞARISIZ: Genel poz değişimi çok düşük - FOTOĞRAF?")
            return False, "Başarısız (Fotograf?)"

        # --- Success Condition ---
        # All required checks completed successfully?
        # Note: Pose variation doesn't need to be OK, just not False (not a photo)
        if blinks_ok and head_move_ok and look_lr_ok:
            self.liveness_passed = True
            logging.info(f"Canlılık kontrolü BAŞARILI (Blink: {blinks_ok}, Kafa Hareketi: {head_move_ok}, Sağa/Sola Bakma: {look_lr_ok})")
            return True, "Geçildi"

        # --- In Progress Status (Not yet Success or Failure) ---
        else:
            # Create detailed status display
            status_parts = []
            status_parts.append(f"B:{self.blinks_detected_count}/{REQUIRED_BLINKS}")  # Blink status
            # Head movement status: ? (unknown), OK (passed), YOK (insufficient)
            hm_status = "?" if self.head_movement_ok is None else ("OK" if self.head_movement_ok else "YOK")
            status_parts.append(f"HM:{hm_status}")
            # Look left/right status: N (none), L (left ok), R (right ok), OK (both ok)
            lr_status = "N"
            if self.looked_left and self.looked_right: lr_status = "OK"
            elif self.looked_left: lr_status = "L->"  # Looked left, waiting for right
            elif self.looked_right: lr_status = "<-R"  # Looked right, waiting for left
            else: lr_status = "<->"  # Waiting for both
            status_parts.append(f"L/R:{lr_status}")
            # Timer status
            status_parts.append(f"{self.liveness_check_frame_counter}/{LIVENESS_TIMEOUT_FRAMES}F")

            return False, f"Kontrol ({', '.join(status_parts)})"

    def increment_frame_counter(self):
        """Increment the frame counter for timeout checking."""
        if self.is_checking:
            self.liveness_check_frame_counter += 1 