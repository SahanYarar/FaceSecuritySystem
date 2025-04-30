#!/usr/bin/env python3
# -- coding: utf-8 --

import cv2
import time
import logging
import numpy as np
import collections
import traceback
import dlib
import sys

# Import the Interface class from the ui module
from ui.interface import Interface
from camera.camera_manager import CameraManager
from face.face_processor import FaceProcessor
from face.liveness_detector import LivenessDetector
from face.face_tracker import FaceTracker
from db.face_storage import FaceStorage
from hardware.door_controller import DoorController
from hardware.door_manager import DoorManager
from utils.helpers import handle_error, eye_aspect_ratio, shape_to_np, calculate_iou
from common.constants import (
     FRAME_WIDTH, PROCESS_WIDTH,
     REQUIRED_BLINKS,HEAD_MOVEMENT_FRAMES, POSE_HISTORY_FRAMES,COLOR_GREEN, COLOR_RED,COLOR_BLUE, COLOR_YELLOW, COLOR_WHITE,
)

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Ana Sistem Sınıfı ---
class DoorSecuritySystem:
    """Tüm alt sistemleri yöneten ana sınıf."""
    def __init__(self):
        logging.info("Face Security System v3.0 initializing...")
        self.cap = None
        self.running = False
        self.initialized_successfully = False

        try:
            # Alt sistemleri başlat
            self.camera_manager = CameraManager()
            self.face_processor = FaceProcessor()
            self.storage = FaceStorage()
            self.door_controller = DoorController()
            self.interface = Interface()

            # Initialize managers
            self.liveness_detector = LivenessDetector()
            self.face_tracker = FaceTracker()
            self.door_manager = DoorManager(self.door_controller)

            # Kamera başlatma
            self.cap = self.camera_manager.init_camera(width=FRAME_WIDTH)
            if not self.cap:
                raise Exception("Camera initialization failed")

            # --- Durum Değişkenleri ---
            self.current_mode = "normal"
            self.input_text = ""
            self.frame_counter = 0

            # Tanıma/Stabilite
            self.candidate_name = None
            self.recognition_streak_count = 0
            self.stable_match_name = None
            self.last_known_rect = None
            self.last_processed_landmarks = None
            self.scale_factor = 1.0
            self.last_detection_time = 0
            self.current_process_frame_size = (0, 0) # (H, W)

            # Canlılık Değişkenleri (Blink, Centroid, Pose Var, Look L/R)
            self.liveness_passed = False
            self.ear_consec_counter = 0
            self.liveness_check_frame_counter = 0
            self.blinks_detected_count = 0
            self.centroid_history = collections.deque(maxlen=HEAD_MOVEMENT_FRAMES)
            self.head_movement_ok = None # True, False, None
            self.pose_history = collections.deque(maxlen=POSE_HISTORY_FRAMES)
            self.head_pose_variation_ok = None # Genel poz değişimi (True, False, None)
            self.initial_yaw = None        # YENİ: Canlılık başladığındaki Yaw açısı
            self.looked_left = False       # YENİ: Yeterince sola bakıldı mı?
            self.looked_right = False      # YENİ: Yeterince sağa bakıldı mı?

            # Kapı Durumu
            self.door_opened_time = None

            # UI Durum Sözlüğü
            self.system_status = {
                "status": "Waiting for Face",
                "color": COLOR_BLUE,
                "liveness": "Waiting for Face",
                "liveness_color": COLOR_WHITE,
                "action_handler": self.handle_ui_action
            }

            self.initialized_successfully = True
            logging.info("System initialized successfully")

        except Exception as init_err:
            logging.exception(f"Critical error during initialization: {init_err}")
            self.initialized_successfully = False
            self._cleanup()

    def handle_ui_action(self, action_type, value):
        """Handle UI actions."""
        if action_type == "set_mode":
            if value in ["normal", "register", "delete"]:
                if self.current_mode != value:
                    logging.info(f"Mode changing: {self.current_mode} -> {value}")
                    self.current_mode = value
                    self.input_text = ""
                    self.interface.selected_name_for_delete = None
                    self._reset_state()
            else:
                logging.warning(f"Invalid mode attempted: {value}")
        elif action_type == "quit":
            logging.info("Quit command received")
            self.running = False
        elif action_type == "register_face":
            self._register_face(value)
        elif action_type == "delete_face":
            if value:
                self._delete_face(value)
            else:
                self.interface.set_message("Please select a name to delete!", COLOR_RED)

    def _reset_state(self):
        """Reset system state."""
        self.face_tracker.reset()
        self.liveness_detector.reset()
        self.door_manager.update_status("Waiting for Face", COLOR_BLUE)

    def _process_frame(self, frame):
        """Process incoming frame."""
        try:
            orig_h, orig_w = frame.shape[:2]
            self.face_tracker.scale_factor = PROCESS_WIDTH / orig_w
            process_height = int(orig_h * self.face_tracker.scale_factor)
            self.face_tracker.current_process_frame_size = (process_height, PROCESS_WIDTH)
            process_frame = cv2.resize(frame, (PROCESS_WIDTH, process_height), interpolation=cv2.INTER_AREA)
            gray_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
            return process_frame, gray_frame
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            return None, None

    def _detect_and_process_face(self, gray_frame, process_frame):
        """Detect and process a face in the frame."""
        try:
            detected_faces = self.face_processor.detect_faces(gray_frame)
            if not detected_faces:
                return None, None, None

            # Get the largest face in the frame
            face_rect = max(detected_faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))

            # Get landmarks for the detected face
            landmarks = self.face_processor.get_landmarks(gray_frame, face_rect)
            if landmarks is None:
                return None, None, None

            # Get face descriptor
            face_descriptor = self.face_processor.get_face_descriptor(process_frame, landmarks)
            if face_descriptor is None:
                return None, None, None

            # Convert coordinates to display scale
            display_rect = dlib.rectangle(
                int(face_rect.left() / self.face_tracker.scale_factor),
                int(face_rect.top() / self.face_tracker.scale_factor),
                int(face_rect.right() / self.face_tracker.scale_factor),
                int(face_rect.bottom() / self.face_tracker.scale_factor)
            )

            return display_rect, landmarks, face_descriptor

        except Exception as e:
            handle_error(f"Face detection error: {e}", "Face Detection")
            return None, None, None

    def _update_liveness(self, landmarks):
        """Update liveness detection state."""
        if not (self.current_mode == "normal" and landmarks is not None):
            return

        # Update EAR-based liveness
        ear_value = self.face_processor.check_liveness_ear(landmarks)
        self.liveness_detector.update_ear(ear_value)

        # Update centroid-based liveness
        centroid = self.face_processor.get_face_centroid(landmarks)
        hm_check_result = self.face_processor.check_liveness_head_movement(self.liveness_detector.centroid_history)
        self.liveness_detector.update_centroid(centroid, hm_check_result)

        # Update pose-based liveness
        pose_angles = self.face_processor.get_head_pose_angles(landmarks, self.face_tracker.current_process_frame_size)
        current_yaw = pose_angles[1] if pose_angles else None
        pv_check_result = self.face_processor.check_liveness_pose_variation(self.liveness_detector.pose_history)
        self.liveness_detector.update_pose(pose_angles, pv_check_result, current_yaw)

        # Check liveness status
        liveness_passed, liveness_data = self.liveness_detector.check_liveness()
        
        # Update interface with liveness data
        self.interface.update_liveness(liveness_data)
        
        # Update system status based on liveness
        if liveness_passed:
            self.system_status["status"] = f"Liveness passed for {self.face_tracker.stable_match_name}"
            self.system_status["color"] = COLOR_GREEN
        elif liveness_data.get("status") == "in_progress":
            self.system_status["status"] = f"Checking liveness for {self.face_tracker.stable_match_name}"
            self.system_status["color"] = COLOR_YELLOW
        elif liveness_data.get("status") == "expired":
            self.system_status["status"] = f"Liveness expired for {self.face_tracker.stable_match_name}"
            self.system_status["color"] = COLOR_RED

        self.liveness_detector.increment_frame_counter()

        # Update door manager status
        self.door_manager.update_status(liveness_data.get("status", "unknown"), 
                                      self.interface.liveness_color)

        # If liveness passed, update door state
        if liveness_passed and self.face_tracker.stable_match_name:
            logging.info(f"Liveness passed for {self.face_tracker.stable_match_name}")
            self.door_manager.update_door_state(
                True,  # is_stable_now
                True,  # liveness_passed
                self.current_mode,
                self.face_tracker.stable_match_name
            )

    def _draw_frame_elements(self, display_frame):
        """Draw UI elements on the frame."""
        if self.face_tracker.last_known_rect:
            x1, y1, x2, y2 = (self.face_tracker.last_known_rect.left(),
                             self.face_tracker.last_known_rect.top(),
                             self.face_tracker.last_known_rect.right(),
                             self.face_tracker.last_known_rect.bottom())
            
            # Get current recognition percentage
            best_match_name = None
            best_match_percentage = 0.0
            if self.face_tracker.candidate_name:
                for name, known_descriptor in self.storage.get_known_faces().items():
                    if name == self.face_tracker.candidate_name:
                        percentage = self.face_processor.compare_faces(
                            self.face_tracker.last_known_descriptor, 
                            known_descriptor
                        )
                        best_match_percentage = percentage
                        best_match_name = name
                        break

            # Draw face rectangle
            color = COLOR_GREEN if self.liveness_detector.liveness_passed else COLOR_YELLOW
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Prepare debug information
            debug_info = {}
            
            # Recognition info
            if self.face_tracker.stable_match_name:
                debug_info["Name"] = f"{self.face_tracker.stable_match_name} ({best_match_percentage:.1f}%)"
            elif self.face_tracker.candidate_name:
                debug_info["Name"] = f"{self.face_tracker.candidate_name} ({best_match_percentage:.1f}%)"
            else:
                debug_info["Name"] = "Unknown"


            # Draw debug information
            y_offset = y1 - 10
            for key, value in debug_info.items():
                cv2.putText(display_frame, f"{key}: {value}", (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
                y_offset += 30

        try:
            known_names = list(self.storage.get_known_faces().keys())
            # Combine system status with door manager status
            system_status = {
                "status": self.system_status["status"],
                "color": self.system_status["color"],
                "liveness": self.system_status["liveness"],
                "liveness_color": self.system_status["liveness_color"],
                "action_handler": self.handle_ui_action,
                "door_remaining_time": self.door_manager.system_status["door_remaining_time"]
            }
            self.interface.draw_ui(display_frame, self.current_mode, self.input_text, known_names, system_status)
        except Exception as e:
            logging.error(f"UI drawing error: {e}")
            traceback.print_exc()

    def run(self):
        """Main application loop."""
        if not self.initialized_successfully:
            logging.error("System not initialized, cannot run")
            return
        if not self.cap or not self.cap.isOpened():
            logging.error("Camera not available or cannot be opened")
            return

        window_name = "Face Security System v3.0"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window_name, self.handle_mouse_event)
        except Exception as e:
            logging.error(f"OpenCV window creation failed: {e}")
            return

        self.running = True
        logging.info("Main loop started. Press 'q' to quit")
        self.handle_ui_action("set_mode", "normal")

        # Add state tracking variables
        last_status = None
        status_update_time = time.time()
        min_status_update_interval = 0.5  # Minimum time between status updates

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logging.warning("Failed to get frame from camera, waiting...")
                    time.sleep(0.1)
                    continue

                display_frame = frame.copy()
                self.frame_counter += 1

                process_frame, gray_frame = self._process_frame(frame)
                if process_frame is None or gray_frame is None:
                    continue

                # Detect and process face
                face_rect, landmarks, face_descriptor = self._detect_and_process_face(gray_frame, process_frame)
                current_time = time.time()

                if self.current_mode == "normal":
                    if face_rect is not None and landmarks is not None and face_descriptor is not None:
                        # Update face tracking
                        if not self.face_tracker.update_tracking(face_rect):
                            logging.warning("Face tracking lost")
                            self.face_tracker.reset()
                            self.liveness_detector.reset()
                            continue

                        # Update face recognition
                        if face_descriptor is not None:
                            # Get current recognition percentage
                            best_match_name = None
                            best_match_percentage = 0.0
                            for name, known_descriptor in self.storage.get_known_faces().items():
                                percentage = self.face_processor.compare_faces(face_descriptor, known_descriptor)
                                if percentage > best_match_percentage:
                                    best_match_percentage = percentage
                                    best_match_name = name

                            # Update recognition status
                            if best_match_percentage > 45.0:
                                self.face_tracker.update_recognition(face_descriptor, self.storage.get_known_faces(), self.face_processor)
                                if self.face_tracker.stable_match_name:
                                    self.system_status["status"] = f"Recognized: {self.face_tracker.stable_match_name})"
                                    self.system_status["color"] = COLOR_GREEN
                                    # Start liveness check if not already checking
                                    if not self.liveness_detector.is_checking:
                                        self.liveness_detector.start_checking()
                                else:
                                    self.system_status["status"] = f"Verifying: {best_match_name} ({best_match_percentage:.1f}%)"
                                    self.system_status["color"] = COLOR_YELLOW
                            else:
                                self.system_status["status"] = f"No match found ({best_match_percentage:.1f}%)"
                                self.system_status["color"] = COLOR_RED

                        # Update liveness detection
                        self._update_liveness(landmarks)

                        # Check if we should open the door
                        if (self.face_tracker.stable_match_name and 
                            self.liveness_detector.liveness_passed and 
                            not self.door_manager.get_state()):
                            self.door_manager.open_door(self.face_tracker.stable_match_name)
                            
                            # Update system status to show door is open
                            self.system_status["status"] = f"Door opened for {self.face_tracker.stable_match_name}"
                            self.system_status["color"] = COLOR_GREEN
                            
                            # Get current liveness data
                            _, liveness_data = self.liveness_detector.check_liveness()
                            self.interface.update_liveness(liveness_data)

                    else:
                        # No face detected - only reset if we're not in a passed state
                        if not self.liveness_detector.liveness_passed:
                            self.face_tracker.reset()
                            self.liveness_detector.reset()
                            self.system_status["status"] = "No face detected"
                            self.system_status["color"] = COLOR_RED
                            # Reset liveness display
                            self.interface.update_liveness(None)
                        else:
                            # If liveness is passed, keep showing the status
                            _, liveness_data = self.liveness_detector.check_liveness()
                            self.interface.update_liveness(liveness_data)

                    # Only update UI if status has changed or enough time has passed
                    if (self.system_status["status"] != last_status or 
                        current_time - status_update_time >= min_status_update_interval):
                        self.interface.update_status(self.system_status["status"], self.system_status["color"])
                        last_status = self.system_status["status"]
                        status_update_time = current_time

                self._draw_frame_elements(display_frame)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    if self.current_mode == "normal":
                        logging.info("'q' pressed, quitting")
                        self.running = False
                    else:
                        logging.info("'q' pressed, returning to normal mode")
                        self.handle_ui_action("set_mode", "normal")
                elif self.current_mode == "normal":
                    if key == ord('r'):
                        self.handle_ui_action("set_mode", "register")
                    elif key == ord('d'):
                        self.handle_ui_action("set_mode", "delete")
                elif self.current_mode in ["register", "delete"]:
                    self._handle_text_input(key)

            except KeyboardInterrupt:
                logging.info("CTRL+C detected, quitting")
                self.running = False
            except Exception as loop_err:
                logging.exception(f"Unexpected error in main loop: {loop_err}")
                time.sleep(1)

        self._cleanup()

    def handle_mouse_event(self, event, x, y, flags, param):
        """Fare olaylarını (sol tıklama) işler ve arayüze iletir."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.interface.handle_click(x, y)

    def _handle_text_input(self, key):
        """Handle text input in register and delete modes."""
        if key == 27:  # ESC
            self.handle_ui_action("set_mode", "normal")
        elif key == 13:  # Enter
            if self.current_mode == "register":
                self._register_face(self.input_text)
        elif key in [8, 127]:  # Backspace / Delete
            self.input_text = self.input_text[:-1]
        elif 32 <= key <= 126:  # Printable characters
            if len(self.input_text) < 20:
                self.input_text += chr(key)
            else:
                self.interface.set_message("Maximum name length!", COLOR_YELLOW, duration=2.0)

    def _register_face(self, name):
        """Register a new face."""
        if self.current_mode != "register":
            logging.warning("Face registration only available in register mode")
            return

        name = name.strip()
        if not name:
            self.interface.set_message("Please enter a valid name!", COLOR_RED)
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.interface.set_message("Registration: Failed to get camera frame!", COLOR_RED)
            return

        process_frame, gray_frame = self._process_frame(frame)
        if process_frame is None or gray_frame is None:
            self.interface.set_message("Registration: Frame processing failed!", COLOR_RED)
            return

        faces = self.face_processor.detect_faces(gray_frame)
        if not faces:
            self.interface.set_message("Registration: No face detected!", COLOR_RED)
            return
        if len(faces) > 1:
            self.interface.set_message("Registration: Only one face allowed!", COLOR_RED)
            return

        landmarks = self.face_processor.get_landmarks(gray_frame, faces[0])
        if not landmarks:
            self.interface.set_message("Registration: Failed to get landmarks!", COLOR_RED)
            return

        descriptor = self.face_processor.get_face_descriptor(process_frame, landmarks)
        if descriptor is None:
            self.interface.set_message("Registration: Failed to compute descriptor!", COLOR_RED)
            return

        if self.storage.add_face(name, descriptor):
            self.interface.set_message(f"'{name}' successfully registered!", COLOR_GREEN, duration=5.0)
            self.handle_ui_action("set_mode", "normal")
        else:
            self.interface.set_message(f"Failed to register '{name}'!", COLOR_RED)

    def _delete_face(self, name):
        """Delete a registered face."""
        if self.current_mode != "delete":
            logging.warning("Face deletion only available in delete mode")
            return
        if not name:
            self.interface.set_message("Please select a name to delete!", COLOR_RED)
            return

        if self.storage.delete_face(name):
            self.interface.set_message(f"'{name}' successfully deleted.", COLOR_GREEN)
            self.interface.selected_name_for_delete = None
            self.door_manager.update_status("Delete: Select name or Cancel", COLOR_YELLOW)
        else:
            self.interface.set_message(f"Failed to delete '{name}'!", COLOR_RED)

    def _cleanup(self):
        """Clean up system resources."""
        logging.info("Starting cleanup...")
        self.running = False
        time.sleep(0.1)

        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            try:
                self.cap.release()
                logging.info("Camera released")
            except Exception as e:
                logging.warning(f"Camera release error: {e}")
        self.cap = None

        try:
            cv2.destroyAllWindows()
            logging.info("OpenCV windows closed")
        except Exception as e:
            logging.warning(f"OpenCV window closing error: {e}")

        self.door_manager.cleanup()
        logging.info("Cleanup completed")

# --- Program Başlangıcı ---
if __name__ == "__main__":
    system = None
    try:
        # Sistemi başlat
        system = DoorSecuritySystem()
        # Başlatma başarılıysa çalıştır
        if system.initialized_successfully:
             system.run()
        else:
             logging.error("Sistem başlatılamadığı için çalıştırılamıyor.")
    except KeyboardInterrupt:
        logging.info("CTRL+C ile program sonlandırıldı.")
    except Exception as main_err:
        # Ana kapsamda beklenmedik bir hata olursa logla
        logging.exception(f"Programın ana kısmında beklenmedik ve yakalanmayan hata: {main_err}")
    finally:
        # Program sonlanırken (normal veya hata ile) temizliği çağır
        if system:
            logging.info("Program sonlanıyor, temizlik yapılıyor...")
            system._cleanup()
        else:
             logging.info("Sistem nesnesi oluşturulamadığı için temizlik atlandı.")
        logging.info("Program sonlandı.")