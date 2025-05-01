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
     REQUIRED_BLINKS, HEAD_MOVEMENT_FRAMES, POSE_HISTORY_FRAMES,
     COLOR_GREEN, COLOR_RED, COLOR_BLUE, COLOR_YELLOW, COLOR_WHITE,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class DoorSecuritySystem:
    """
    Main security system class that coordinates all components:
    1. Face Detection and Recognition
    2. Liveness Detection
    3. Door Control
    4. User Interface
    
    System Flow:
    1. Initialize all components (camera, face processor, etc.)
    2. Process each frame:
       - Detect faces
       - Extract landmarks
       - Compute face descriptors
       - Check liveness
       - Update recognition state
    3. Control door based on recognition and liveness
    4. Update UI with system status
    """
    def __init__(self):
        logging.info("Face Security System v3.0 initializing...")
        self.cap = None
        self.running = False
        self.initialized_successfully = False

        try:
            # Initialize all system components
            # 1. Camera Manager: Handles camera initialization and frame capture
            #    - Sets resolution and frame rate
            #    - Handles camera errors and reconnection
            self.camera_manager = CameraManager()
            
            # 2. Face Processor: Core face recognition component
            #    - Face detection using dlib's HOG detector
            #    - 68-point facial landmark detection
            #    - 128D face descriptor computation
            self.face_processor = FaceProcessor()
            
            # 3. Face Storage: Database for known faces
            #    - Stores face descriptors
            #    - Handles face registration and deletion
            self.storage = FaceStorage()
            
            # 4. Door Controller: Hardware interface
            #    - Controls door lock/unlock
            #    - Handles door status monitoring
            self.door_controller = DoorController()
            
            # 5. User Interface: System UI
            #    - Displays camera feed
            #    - Shows system status
            #    - Handles user input
            self.interface = Interface()

            # Initialize specialized components
            # 1. Liveness Detector: Anti-spoofing component
            #    - Eye blink detection
            #    - Head movement analysis
            #    - Pose variation checking
            self.liveness_detector = LivenessDetector()
            
            # 2. Face Tracker: Maintains face state across frames
            #    - Tracks face position using IoU
            #    - Maintains recognition streak
            #    - Handles face loss and timeout
            self.face_tracker = FaceTracker()
            
            # 3. Door Manager: High-level door control
            #    - Manages door state
            #    - Handles access control logic
            self.door_manager = DoorManager(self.door_controller)

            # Initialize camera with specified width
            # - FRAME_WIDTH: Width of captured frames
            # - PROCESS_WIDTH: Width for face processing (smaller for performance)
            self.cap = self.camera_manager.init_camera(width=FRAME_WIDTH)
            if not self.cap:
                raise Exception("Camera initialization failed")

            # System state variables
            self.current_mode = "normal"  # Current system mode (normal/register/delete)
            self.input_text = ""  # User input text
            self.frame_counter = 0  # Frame counter for timing

            # Face recognition state
            self.candidate_name = None  # Current best match
            self.recognition_streak_count = 0  # Consecutive frames with same match
            self.stable_match_name = None  # Confirmed identity
            self.last_known_rect = None  # Last known face position
            self.last_processed_landmarks = None  # Last detected landmarks
            self.scale_factor = 1.0  # Scale between original and processed frame
            self.last_detection_time = 0  # Timestamp of last detection
            self.current_process_frame_size = (0, 0)  # Size of processed frame

            # Liveness detection state
            self.liveness_passed = False  # Liveness check status
            self.ear_consec_counter = 0  # Consecutive frames with low EAR
            self.liveness_check_frame_counter = 0  # Frame counter for liveness
            self.blinks_detected_count = 0  # Number of detected blinks
            self.centroid_history = collections.deque(maxlen=HEAD_MOVEMENT_FRAMES)  # Head movement history
            self.head_movement_ok = None  # Head movement check status
            self.pose_history = collections.deque(maxlen=POSE_HISTORY_FRAMES)  # Pose history
            self.head_pose_variation_ok = None  # Pose variation check status
            self.initial_yaw = None  # Initial head yaw angle
            self.looked_left = False  # Left look check
            self.looked_right = False  # Right look check

            # Door state
            self.door_opened_time = None  # Timestamp of last door open

            # System status dictionary
            self.system_status = {
                "status": "Waiting for Face",  # Current system status
                "color": COLOR_BLUE,  # Status color
                "liveness": "Waiting for Face",  # Liveness status
                "liveness_color": COLOR_WHITE,  # Liveness status color
                "action_handler": self.handle_ui_action  # UI action handler
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
        """
        Process incoming frame for face detection and recognition.
        
        Process:
        1. Resize frame for processing
        2. Convert to grayscale
        3. Apply scale factor for face detection
        
        Args:
            frame: Original BGR frame from camera
            
        Returns:
            tuple: (process_frame, gray_frame)
            - process_frame: Resized frame for processing
            - gray_frame: Grayscale version for face detection
        """
        try:
            # Get original frame dimensions
            orig_h, orig_w = frame.shape[:2]
            
            # Calculate scale factor for processing
            self.face_tracker.scale_factor = PROCESS_WIDTH / orig_w
            
            # Calculate new height maintaining aspect ratio
            process_height = int(orig_h * self.face_tracker.scale_factor)
            
            # Update processing frame size
            self.face_tracker.current_process_frame_size = (process_height, PROCESS_WIDTH)
            
            # Resize frame for processing
            process_frame = cv2.resize(frame, (PROCESS_WIDTH, process_height), 
                                     interpolation=cv2.INTER_AREA)
            
            # Convert to grayscale for face detection
            gray_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
            
            return process_frame, gray_frame
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            return None, None

    def _detect_and_process_face(self, gray_frame, process_frame):
        """
        Detect and process a face in the frame.
        
        Process:
        1. Detect faces using HOG detector
        2. Get facial landmarks
        3. Compute face descriptor
        4. Convert coordinates to display scale
        
        Args:
            gray_frame: Grayscale frame for face detection
            process_frame: Color frame for descriptor computation
            
        Returns:
            tuple: (display_rect, landmarks, face_descriptor)
            - display_rect: Face rectangle in display coordinates
            - landmarks: 68 facial landmarks
            - face_descriptor: 128D face descriptor
        """
        try:
            # Detect faces in grayscale frame
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
        """
        Update liveness detection state based on facial landmarks.
        
        Process:
        1. Check if in normal mode and landmarks are available
        2. Update EAR-based liveness (eye blink detection)
        3. Update centroid-based liveness (head movement)
        4. Update pose-based liveness (head rotation)
        5. Check overall liveness status
        6. Update system status and door state
        
        Args:
            landmarks: 68 facial landmarks from dlib
            
        Side Effects:
            - Updates liveness detector state
            - Updates system status
            - Updates door manager state
        """
        if not (self.current_mode == "normal" and landmarks is not None):
            return

        # 1. Update EAR-based liveness
        # - EAR (Eye Aspect Ratio) measures eye openness
        # - Low EAR indicates closed eyes (blink)
        # - Consecutive low EAR frames indicate valid blink
        ear_value = self.face_processor.check_liveness_ear(landmarks)
        self.liveness_detector.update_ear(ear_value)

        # 2. Update centroid-based liveness
        # - Calculate face centroid from landmarks
        # - Track centroid movement over time
        # - Check if movement is within valid range
        centroid = self.face_processor.get_face_centroid(landmarks)
        hm_check_result = self.face_processor.check_liveness_head_movement(
            self.liveness_detector.centroid_history)
        self.liveness_detector.update_centroid(centroid, hm_check_result)

        # 3. Update pose-based liveness
        # - Calculate head pose angles (pitch, yaw, roll)
        # - Track pose variation over time
        # - Check for valid head rotations
        pose_angles = self.face_processor.get_head_pose_angles(
            landmarks, self.face_tracker.current_process_frame_size)
        current_yaw = pose_angles[1] if pose_angles else None
        pv_check_result = self.face_processor.check_liveness_pose_variation(
            self.liveness_detector.pose_history)
        self.liveness_detector.update_pose(pose_angles, pv_check_result, current_yaw)

        # 4. Check overall liveness status
        # - Combines results from all liveness checks
        # - Updates system status based on result
        liveness_passed, liveness_data = self.liveness_detector.check_liveness()
        
        # 5. Update interface with liveness data
        self.interface.update_liveness(liveness_data)
        
        # 6. Update system status based on liveness
        if liveness_passed:
            self.system_status["status"] = f"Liveness passed for {self.face_tracker.stable_match_name}"
            self.system_status["color"] = COLOR_GREEN
        elif liveness_data.get("status") == "in_progress":
            self.system_status["status"] = f"Checking liveness for {self.face_tracker.stable_match_name}"
            self.system_status["color"] = COLOR_YELLOW
        elif liveness_data.get("status") == "expired":
            self.system_status["status"] = f"Liveness expired for {self.face_tracker.stable_match_name}"
            self.system_status["color"] = COLOR_RED

        # 7. Increment frame counter for timing
        self.liveness_detector.increment_frame_counter()

        # 8. Update door manager status
        self.door_manager.update_status(
            liveness_data.get("status", "unknown"), 
            self.interface.liveness_color)

        # 9. Update door state if liveness passed
        if liveness_passed and self.face_tracker.stable_match_name:
            logging.info(f"Liveness passed for {self.face_tracker.stable_match_name}")
            self.door_manager.update_door_state(
                True,  # is_stable_now
                True,  # liveness_passed
                self.current_mode,
                self.face_tracker.stable_match_name
            )

    def _draw_frame_elements(self, display_frame):
        """
        Draw UI elements and face information on the display frame.
        
        Process:
        1. Draw face rectangle if face is detected
        2. Calculate recognition percentage
        3. Draw debug information
        4. Update UI with system status
        
        Args:
            display_frame: Frame to draw on
            
        Side Effects:
            - Updates display frame with visual elements
            - Updates UI status
        """
        if self.face_tracker.last_known_rect:
            # 1. Get face rectangle coordinates
            x1, y1, x2, y2 = (self.face_tracker.last_known_rect.left(),
                             self.face_tracker.last_known_rect.top(),
                             self.face_tracker.last_known_rect.right(),
                             self.face_tracker.last_known_rect.bottom())
            
            # 2. Calculate current recognition percentage
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

            # 3. Draw face rectangle with appropriate color
            color = COLOR_GREEN if self.liveness_detector.liveness_passed else COLOR_YELLOW
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # 4. Prepare debug information
            debug_info = {}
            
            # 5. Add recognition info to debug display
            if self.face_tracker.stable_match_name:
                debug_info["Name"] = f"{self.face_tracker.stable_match_name} ({best_match_percentage:.1f}%)"
            elif self.face_tracker.candidate_name:
                debug_info["Name"] = f"{self.face_tracker.candidate_name} ({best_match_percentage:.1f}%)"
            else:
                debug_info["Name"] = "Unknown"

            # 6. Draw debug information on frame
            y_offset = y1 - 10
            for key, value in debug_info.items():
                cv2.putText(display_frame, f"{key}: {value}", (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
                y_offset += 30

        try:
            # 7. Get list of known faces for UI
            known_names = list(self.storage.get_known_faces().keys())
            
            # 8. Prepare system status for UI
            system_status = {
                "status": self.system_status["status"],
                "color": self.system_status["color"],
                "liveness": self.system_status["liveness"],
                "liveness_color": self.system_status["liveness_color"],
                "action_handler": self.handle_ui_action,
                "door_remaining_time": self.door_manager.system_status["door_remaining_time"]
            }
            
            # 9. Draw UI elements
            self.interface.draw_ui(display_frame, self.current_mode, 
                                 self.input_text, known_names, system_status)
        except Exception as e:
            logging.error(f"UI drawing error: {e}")
            traceback.print_exc()

    def run(self):
        """
        Main application loop that handles the entire face recognition system.
        
        Process:
        1. Initialize system and check prerequisites
        2. Create OpenCV window and set up mouse callback
        3. Main processing loop:
           - Capture and process frames
           - Detect and recognize faces
           - Update liveness detection
           - Handle user input
           - Update UI
        4. Clean up resources on exit
        
        Side Effects:
            - Creates and manages OpenCV window
            - Updates system state continuously
            - Handles user input and mode changes
            - Controls door access
        """
        # 1. Check system initialization
        if not self.initialized_successfully:
            logging.error("System not initialized, cannot run")
            return
        if not self.cap or not self.cap.isOpened():
            logging.error("Camera not available or cannot be opened")
            return

        # 2. Create OpenCV window
        window_name = "Face Security System v3.0"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window_name, self.handle_mouse_event)
        except Exception as e:
            logging.error(f"OpenCV window creation failed: {e}")
            return

        # 3. Initialize system state
        self.running = True
        logging.info("Main loop started. Press 'q' to quit")
        self.handle_ui_action("set_mode", "normal")

        # 4. Initialize status tracking
        last_status = None
        status_update_time = time.time()
        min_status_update_interval = 0.5  # Minimum time between status updates

        # 5. Main processing loop
        while self.running:
            try:
                # 5.1 Capture frame from camera
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logging.warning("Failed to get frame from camera, waiting...")
                    time.sleep(0.1)
                    continue

                # 5.2 Prepare display frame
                display_frame = frame.copy()
                self.frame_counter += 1

                # 5.3 Process frame for face detection
                process_frame, gray_frame = self._process_frame(frame)
                if process_frame is None or gray_frame is None:
                    continue

                # 5.4 Detect and process face
                face_rect, landmarks, face_descriptor = self._detect_and_process_face(
                    gray_frame, process_frame)
                current_time = time.time()

                # 5.5 Handle face recognition in normal mode
                if self.current_mode == "normal":
                    if face_rect is not None and landmarks is not None and face_descriptor is not None:
                        # 5.5.1 Update face tracking
                        if not self.face_tracker.update_tracking(face_rect):
                            logging.warning("Face tracking lost")
                            self.face_tracker.reset()
                            self.liveness_detector.reset()
                            continue

                        # 5.5.2 Update face recognition
                        if face_descriptor is not None:
                            # Find best matching face
                            best_match_name = None
                            best_match_percentage = 0.0
                            for name, known_descriptor in self.storage.get_known_faces().items():
                                percentage = self.face_processor.compare_faces(
                                    face_descriptor, known_descriptor)
                                if percentage > best_match_percentage:
                                    best_match_percentage = percentage
                                    best_match_name = name

                            # Update recognition status
                            if best_match_percentage > 45.0:
                                self.face_tracker.update_recognition(
                                    face_descriptor, 
                                    self.storage.get_known_faces(), 
                                    self.face_processor)
                                
                                if self.face_tracker.stable_match_name:
                                    self.system_status["status"] = f"Recognized: {self.face_tracker.stable_match_name}"
                                    self.system_status["color"] = COLOR_GREEN
                                    if not self.liveness_detector.is_checking:
                                        self.liveness_detector.start_checking()
                                else:
                                    self.system_status["status"] = f"Verifying: {best_match_name} ({best_match_percentage:.1f}%)"
                                    self.system_status["color"] = COLOR_YELLOW
                            else:
                                self.system_status["status"] = f"No match found ({best_match_percentage:.1f}%)"
                                self.system_status["color"] = COLOR_RED

                        # 5.5.3 Update liveness detection
                        self._update_liveness(landmarks)

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

                    # Update door state even when no face is detected
                    self.door_manager.update_door_state(
                        self.face_tracker.stable_match_name is not None,
                        self.liveness_detector.liveness_passed,
                        self.current_mode,
                        self.face_tracker.stable_match_name
                    )

                    # Only update UI if status has changed or enough time has passed
                    if (self.system_status["status"] != last_status or 
                        current_time - status_update_time >= min_status_update_interval):
                        self.interface.update_status(
                            self.system_status["status"], 
                            self.system_status["color"])
                        last_status = self.system_status["status"]
                        status_update_time = current_time

                # 5.6 Draw UI elements
                self._draw_frame_elements(display_frame)
                cv2.imshow(window_name, display_frame)

                # 5.7 Handle keyboard input
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

        # 6. Clean up resources
        self._cleanup()

    def handle_mouse_event(self, event, x, y, flags, param):
        """
        Handle mouse events in the OpenCV window.
        
        Process:
        1. Check for left mouse button click
        2. Forward click coordinates to UI handler
        
        Args:
            event: OpenCV mouse event type
            x: X coordinate of mouse click
            y: Y coordinate of mouse click
            flags: Additional event flags
            param: Additional parameters
            
        Side Effects:
            - Updates UI based on click location
            - May trigger mode changes or face selection
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.interface.handle_click(x, y)

    def _handle_text_input(self, key):
        """
        Handle text input in register and delete modes.
        
        Process:
        1. Handle special keys (ESC, Enter, Backspace)
        2. Handle printable characters
        3. Update input text and UI
        
        Args:
            key: ASCII value of pressed key
            
        Side Effects:
            - Updates input text
            - Updates UI messages
            - May trigger face registration or deletion
        """
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
        """
        Register a new face in the system.
        
        Process:
        1. Validate input and mode
        2. Capture and process frame
        3. Detect and validate face
        4. Extract landmarks and descriptor
        5. Store in database
        
        Args:
            name: Name to associate with the face
            
        Returns:
            None
            
        Side Effects:
            - Updates face database
            - Updates UI status
            - Changes system mode
        """
        if self.current_mode != "register":
            logging.warning("Face registration only available in register mode")
            return

        # 1. Validate input name
        name = name.strip()
        if not name:
            self.interface.set_message("Please enter a valid name!", COLOR_RED)
            return

        # 2. Capture frame from camera
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.interface.set_message("Registration: Failed to get camera frame!", COLOR_RED)
            return

        # 3. Process frame for face detection
        process_frame, gray_frame = self._process_frame(frame)
        if process_frame is None or gray_frame is None:
            self.interface.set_message("Registration: Frame processing failed!", COLOR_RED)
            return

        # 4. Detect faces
        faces = self.face_processor.detect_faces(gray_frame)
        if not faces:
            self.interface.set_message("Registration: No face detected!", COLOR_RED)
            return
        if len(faces) > 1:
            self.interface.set_message("Registration: Only one face allowed!", COLOR_RED)
            return

        # 5. Get facial landmarks
        landmarks = self.face_processor.get_landmarks(gray_frame, faces[0])
        if not landmarks:
            self.interface.set_message("Registration: Failed to get landmarks!", COLOR_RED)
            return

        # 6. Compute face descriptor
        descriptor = self.face_processor.get_face_descriptor(process_frame, landmarks)
        if descriptor is None:
            self.interface.set_message("Registration: Failed to compute descriptor!", COLOR_RED)
            return

        # 7. Store face in database
        if self.storage.add_face(name, descriptor):
            self.interface.set_message(f"'{name}' successfully registered!", COLOR_GREEN, duration=5.0)
            self.handle_ui_action("set_mode", "normal")
        else:
            self.interface.set_message(f"Failed to register '{name}'!", COLOR_RED)

    def _delete_face(self, name):
        """
        Delete a registered face from the system.
        
        Process:
        1. Validate input and mode
        2. Remove face from database
        3. Update UI and system state
        
        Args:
            name: Name of the face to delete
            
        Returns:
            None
            
        Side Effects:
            - Updates face database
            - Updates UI status
            - Resets selection state
        """
        if self.current_mode != "delete":
            logging.warning("Face deletion only available in delete mode")
            return
        if not name:
            self.interface.set_message("Please select a name to delete!", COLOR_RED)
            return

        # 1. Remove face from database
        if self.storage.delete_face(name):
            self.interface.set_message(f"'{name}' successfully deleted.", COLOR_GREEN)
            self.interface.selected_name_for_delete = None
            self.door_manager.update_status("Delete: Select name or Cancel", COLOR_YELLOW)
        else:
            self.interface.set_message(f"Failed to delete '{name}'!", COLOR_RED)

    def _cleanup(self):
        """
        Clean up system resources and shut down components.
        
        Process:
        1. Stop main loop
        2. Release camera
        3. Close OpenCV windows
        4. Clean up door manager
        5. Log cleanup completion
        
        Side Effects:
            - Releases hardware resources
            - Closes windows and connections
            - Resets system state
        """
        logging.info("Starting cleanup...")
        self.running = False
        time.sleep(0.1)

        # 1. Release camera
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            try:
                self.cap.release()
                logging.info("Camera released")
            except Exception as e:
                logging.warning(f"Camera release error: {e}")
        self.cap = None

        # 2. Close OpenCV windows
        try:
            cv2.destroyAllWindows()
            logging.info("OpenCV windows closed")
        except Exception as e:
            logging.warning(f"OpenCV window closing error: {e}")

        # 3. Clean up door manager
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