import cv2
import logging
import numpy as np
import dlib
import math
from utils.helpers import handle_error, eye_aspect_ratio, shape_to_np
from common.constants import (
    SHAPE_PREDICTOR_PATH, FACE_REC_MODEL_PATH, RECOG_DIST_THRESH,
    EAR_THRESHOLD, EAR_CONSEC_FRAMES, REQUIRED_BLINKS,
    HEAD_MOVEMENT_FRAMES, MIN_CENTROID_MOVEMENT, MAX_CENTROID_MOVEMENT,
    CENTROID_LANDMARK_INDICES, POSE_HISTORY_FRAMES, MIN_POSE_STD_DEV_SUM,
    POSE_LANDMARK_INDICES, LOOK_LEFT_RIGHT_ANGLE_THRESH,
    PITCH_RANGE, YAW_RANGE, ROLL_RANGE
)

class FaceProcessor:
    """Yüz algılama, landmark, tanıma ve canlılık (EAR, Centroid, Pose) işlemlerini yönetir."""
    def __init__(self):
        try:
            logging.info("Dlib modelleri yükleniyor...")
            self.detector = dlib.get_frontal_face_detector() # type: ignore
            self.shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH) # type: ignore
            self.recognizer = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH) # type: ignore
            self.distance_threshold = RECOG_DIST_THRESH
            logging.info(f"Dlib modelleri yüklendi. Mesafe Eşiği: {self.distance_threshold:.4f}")
        except Exception as e:
            handle_error(f"Dlib modelleri yüklenirken kritik hata: {e}", "Model Yükleme")
            handle_error("Model dosyalarının doğru yolda olduğundan emin olun!", "Model Yükleme")
            raise # Programın devam etmemesi için hatayı tekrar fırlat

        # Göz landmark indisleri
        (self.l_start, self.l_end) = (42, 48)
        (self.r_start, self.r_end) = (36, 42)

        # Kafa pozu için 3B model noktaları
        self.model_points_3d = np.array([
            (0.0, 0.0, 0.0),             # Nose tip 30
            (0.0, -330.0, -65.0),        # Chin 8
            (-225.0, 170.0, -135.0),     # Left eye corner 36
            (225.0, 170.0, -135.0),      # Right eye corner 45
            (-150.0, -150.0, -125.0),    # Left Mouth corner 48
            (150.0, -150.0, -125.0)      # Right mouth corner 54
        ], dtype=np.float64)

    def detect_faces(self, gray_frame):
        """Gri tonlamalı karede yüzleri algılar."""
        try:
            # 0: no upsampling (faster), 1: upsample once (detects smaller faces, slower)
            return self.detector(gray_frame, 0)
        except Exception as e:
            return handle_error(f"Yüz algılama hatası: {e}", "Yüz Algılama", [])

    def get_landmarks(self, gray_frame, face_rect):
        """Verilen yüz dikdörtgeni için 68 landmark noktasını bulur."""
        try:
            return self.shape_predictor(gray_frame, face_rect)
        except Exception as e:
            return handle_error(f"Landmark alınamadı: {e}", "Landmark Alma", None)

    def get_face_descriptor(self, frame, landmarks):
        """Verilen yüz ve landmarklar için 128D yüz tanımlayıcı vektörünü hesaplar."""
        if landmarks is None:
            return None
        try:
            return np.array(self.recognizer.compute_face_descriptor(frame, landmarks, 1))
        except Exception as e:
            return handle_error(f"Face descriptor alınamadı: {e}", "Descriptor Hesaplama", None)

    def identify_face(self, face_descriptor, known_faces_db):
        """Descriptor'ı bilinen yüzlerle karşılaştırır."""
        if face_descriptor is None or not known_faces_db:
            return None, 0.0
        min_dist = float('inf')
        best_match = None
        for name, known_descriptor in known_faces_db.items():
            if known_descriptor is None or not isinstance(known_descriptor, np.ndarray) or known_descriptor.shape != (128,):
                logging.warning(f"Identify: '{name}' için geçersiz descriptor, atlanıyor.")
                continue
            try:
                d = np.linalg.norm(face_descriptor - known_descriptor)
                if d < min_dist:
                    min_dist = d
                    best_match = name
            except Exception as e:
                logging.error(f"Identify: '{name}' ile mesafe hesaplama hatası: {e}")

        if best_match and min_dist <= self.distance_threshold:
            confidence = max(0.0, (self.distance_threshold - min_dist) / self.distance_threshold) * 100
            return best_match, confidence
        else:
            return None, 0.0

    def calculate_ear(self, eye_landmarks):
        """Calculate the Eye Aspect Ratio (EAR) for a single eye."""
        try:
            # Convert dlib landmarks to numpy array if needed
            if not isinstance(eye_landmarks, np.ndarray):
                eye_landmarks = shape_to_np(eye_landmarks)
            
            # Compute the vertical distances
            v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
            v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
            
            # Compute the horizontal distance
            h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
            
            # Calculate EAR
            ear = (v1 + v2) / (2.0 * h)
            return ear
        except Exception as e:
            logging.error(f"EAR calculation error: {e}")
            return None

    def detect_blink(self, landmarks, blink_data=None):
        """Detect blinks using eye aspect ratio and state tracking."""
        try:
            if landmarks is None:
                return None

            # Convert dlib landmarks to numpy array
            coords = shape_to_np(landmarks)
            
            # Get eye landmarks
            left_eye = coords[self.l_start:self.l_end]
            right_eye = coords[self.r_start:self.r_end]

            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            
            if left_ear is None or right_ear is None:
                return None

            # Calculate average EAR
            ear = (left_ear + right_ear) / 2.0

            # Initialize blink tracking data if not provided
            if blink_data is None:
                blink_data = {
                    "eye_state": "open",
                    "closed_frames": 0,
                    "open_frames": 0,
                    "blink_count": 0,
                    "blink_in_progress": False,
                    "last_ear": ear,
                    "score": 0  
                }

            # Detect eye state change
            if ear < EAR_THRESHOLD:  # Eyes are closed
                if blink_data["eye_state"] == "open":
                    # Just closed eyes, start potential blink
                    blink_data["blink_in_progress"] = True
                
                blink_data["closed_frames"] += 1
                blink_data["open_frames"] = 0
                blink_data["eye_state"] = "closed"
                
                if 1 <= blink_data["closed_frames"] <= EAR_CONSEC_FRAMES:
                    blink_data["blink_count"] += 1
                    blink_data["score"] += 2  # Award points for valid blink
                    logging.info(f"Blink detected! Count: {blink_data['blink_count']}, Score: {blink_data['score']}")
                
            else:  # Eyes are open
                if (blink_data["eye_state"] == "closed" and 
                    blink_data["blink_in_progress"] and
                    1 <= blink_data["closed_frames"] <= EAR_CONSEC_FRAMES):  # Valid blink duration
                    
                    blink_data["blink_count"] += 1
                    blink_data["score"] += 2  # Award points for valid blink
                    logging.info(f"Blink detected! Count: {blink_data['blink_count']}, Score: {blink_data['score']}")
                
                blink_data["open_frames"] += 1
                blink_data["closed_frames"] = 0
                blink_data["eye_state"] = "open"
                blink_data["blink_in_progress"] = False

            # Penalize continuous eye closure
            if blink_data["closed_frames"] > EAR_CONSEC_FRAMES * 2:
                blink_data["blink_in_progress"] = False
                blink_data["score"] = max(0, blink_data["score"] - 1)  # Penalize for long eye closure

            # Store current EAR for next frame
            blink_data["last_ear"] = ear

            return blink_data

        except Exception as e:
            logging.error(f"Blink detection error: {e}")
            return None

    def check_liveness_ear(self, shape):
        """Check eye aspect ratio for liveness detection."""
        if shape is None:
            return None

        try:
            # Convert dlib landmarks to numpy array
            coords = shape_to_np(shape)
            
            # Get eye landmarks
            left_eye = coords[self.l_start:self.l_end]
            right_eye = coords[self.r_start:self.r_end]

            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            
            if left_ear is None or right_ear is None:
                return None

            # Calculate average EAR
            ear = (left_ear + right_ear) / 2.0
            return ear

        except Exception as e:
            logging.error(f"EAR calculation error: {e}")
            return None

    def get_face_centroid(self, shape):
        """Belirlenen landmarkların ortalama konumunu (centroid) hesaplar."""
        if shape is None: return None
        try:
            coords = shape_to_np(shape)
            max_idx = max(CENTROID_LANDMARK_INDICES)
            if coords.shape[0] <= max_idx:
                logging.warning("Centroid için yeterli landmark yok.")
                return None
            return np.mean(coords[CENTROID_LANDMARK_INDICES], axis=0)
        except Exception as e:
            logging.error(f"Centroid Hesaplama Hatası: {e}")
            return None

    def check_liveness_head_movement(self, centroid_history):
        """Centroid geçmişine bakarak kafa merkezi hareketi olup olmadığını kontrol eder."""
        min_req = max(2, HEAD_MOVEMENT_FRAMES // 2)
        if len(centroid_history) < min_req:
            return None # Henüz yeterli veri yok
        try:
            history_array = np.array(list(centroid_history))
            if history_array.ndim != 2 or history_array.shape[1] != 2:
                logging.warning("Centroid geçmişi geçersiz formatta.")
                return None
            std_dev = np.std(history_array, axis=0)
            movement_magnitude = np.mean(std_dev)

            if MIN_CENTROID_MOVEMENT <= movement_magnitude <= MAX_CENTROID_MOVEMENT:
                return True  # Yeterli hareket var
            elif movement_magnitude < MIN_CENTROID_MOVEMENT:
                 return False if len(centroid_history) >= HEAD_MOVEMENT_FRAMES else None
            else:
                return None # Çok fazla hareket var, kararsız

        except Exception as e:
            logging.error(f"Kafa Hareketi Analiz Hatası: {e}")
            return None

    def detect_head_turn(self, shape):
        """Detect head turn direction using facial landmarks."""
        if shape is None:
            return None

        try:
            # Convert dlib landmarks to numpy array
            coords = shape_to_np(shape)
            
            # Get key facial points
            nose_tip = coords[30]
            left_eye = coords[36]
            right_eye = coords[45]
            left_mouth = coords[48]
            right_mouth = coords[54]

            # Calculate eye width
            eye_width = np.linalg.norm(right_eye - left_eye)
            
            # Calculate distances from nose to eyes
            nose_to_left_eye = np.linalg.norm(nose_tip - left_eye)
            nose_to_right_eye = np.linalg.norm(nose_tip - right_eye)
            
            # Calculate distances from nose to mouth corners
            nose_to_left_mouth = np.linalg.norm(nose_tip - left_mouth)
            nose_to_right_mouth = np.linalg.norm(nose_tip - right_mouth)

            # Calculate ratios
            left_ratio = nose_to_left_eye / eye_width
            right_ratio = nose_to_right_eye / eye_width
            left_mouth_ratio = nose_to_left_mouth / eye_width
            right_mouth_ratio = nose_to_right_mouth / eye_width

            # Debug logging
            logging.debug(f"Left ratio: {left_ratio:.2f}, Right ratio: {right_ratio:.2f}")
            logging.debug(f"Left mouth ratio: {left_mouth_ratio:.2f}, Right mouth ratio: {right_mouth_ratio:.2f}")

            # Determine turn direction
            if left_ratio > 1.2 and left_mouth_ratio > 1.4:  # Looking right
                return "right"
            elif right_ratio > 1.2 and right_mouth_ratio > 1.4:  # Looking left
                return "left"
            elif 0.8 <= left_ratio <= 1.2 and 0.8 <= right_ratio <= 1.2:  # Looking center
                return "center"
            else:
                return None

        except Exception as e:
            logging.error(f"Head turn detection error: {e}")
            return None

    def get_head_pose_angles(self, shape, frame_size):
        """Calculate head pose angles (pitch, yaw, roll) from facial landmarks."""
        if shape is None or frame_size is None:
            return None

        try:
            # First try the simple landmark-based detection
            turn_direction = self.detect_head_turn(shape)
            if turn_direction:
                # Convert turn direction to approximate yaw angle
                if turn_direction == "left":
                    yaw = -45.0
                elif turn_direction == "right":
                    yaw = 45.0
                else:
                    yaw = 0.0
                
                # Use a small random variation for pitch and roll
                pitch = np.random.uniform(-5, 5)
                roll = np.random.uniform(-5, 5)
                
                return (pitch, yaw, roll)

            # Fall back to PnP if simple detection fails
            coords = shape_to_np(shape)
            
            # 3D model points
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])

            # 2D image points
            image_points = np.array([
                coords[30],     # Nose tip
                coords[8],      # Chin
                coords[36],     # Left eye left corner
                coords[45],     # Right eye right corner
                coords[48],     # Left mouth corner
                coords[54]      # Right mouth corner
            ], dtype="double")

            # Camera matrix
            focal_length = frame_size[1]
            center = (frame_size[1]/2, frame_size[0]/2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

            # Distortion coefficients
            dist_coeffs = np.zeros((4,1))

            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE)

            if not success:
                return None

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Extract angles
            pitch, yaw, roll = self._extract_angles(rotation_matrix)

            return (pitch, yaw, roll)

        except Exception as e:
            logging.error(f"Head pose calculation error: {e}")
            return None

    def _extract_angles(self, rotation_matrix):
        """Extract Euler angles from rotation matrix."""
        try:
            # Extract angles
            sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
            
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
                y = np.arctan2(-rotation_matrix[2,0], sy)
                z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
            else:
                x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                y = np.arctan2(-rotation_matrix[2,0], sy)
                z = 0

            # Convert to degrees
            pitch = np.degrees(x)
            yaw = np.degrees(y)
            roll = np.degrees(z)

            return pitch, yaw, roll

        except Exception as e:
            logging.error(f"Angle extraction error: {e}")
            return None, None, None

    def _validate_angles(self, pitch, yaw, roll):
        """Validate if the calculated angles are within reasonable ranges."""
        if pitch is None or yaw is None or roll is None:
            return False

       

        # Check if angles are within ranges
        if not (PITCH_RANGE[0] <= pitch <= PITCH_RANGE[1]):
            logging.debug(f"Invalid pitch angle: {pitch}")
            return False
        if not (YAW_RANGE[0] <= yaw <= YAW_RANGE[1]):
            logging.debug(f"Invalid yaw angle: {yaw}")
            return False
        if not (ROLL_RANGE[0] <= roll <= ROLL_RANGE[1]):
            logging.debug(f"Invalid roll angle: {roll}")
            return False

        return True

    def check_liveness_pose_variation(self, pose_history):
        """Poz geçmişine bakarak yeterli AÇISAL DEĞİŞİM olup olmadığını kontrol eder."""
        min_req = max(3, POSE_HISTORY_FRAMES // 2)
        if len(pose_history) < min_req:
            return None
        try:
            history_array = np.array(list(pose_history))
            if history_array.ndim != 2 or history_array.shape[1] != 3: return None
            std_devs = np.std(history_array, axis=0)
            total_std_dev = np.sum(std_devs)

            if len(pose_history) >= POSE_HISTORY_FRAMES and total_std_dev < MIN_POSE_STD_DEV_SUM:
                return False
            elif total_std_dev >= MIN_POSE_STD_DEV_SUM:
                 return True
            else:
                 return None

        except Exception as e:
            logging.error(f"Poz Değişimi Analiz Hatası: {e}")
            return None

    def compare_faces(self, face_descriptor1, face_descriptor2):
        """Compare two face descriptors and return a similarity percentage."""
        if face_descriptor1 is None or face_descriptor2 is None:
            return 0.0  # Return 0% similarity for invalid descriptors
        
        try:
            # Calculate Euclidean distance
            distance = np.linalg.norm(face_descriptor1 - face_descriptor2)
            
            # Convert distance to percentage (0-100)
            # The maximum possible distance is around 1.0, so we use that as reference
            percentage = max(0.0, (1.0 - distance) * 100)
            return percentage
        except Exception as e:
            logging.error(f"Face comparison error: {e}")
            return 0.0

    def verify_face(self, frame, face_rect, landmarks, known_faces):
        """Verify a face against known faces and return the best match with percentage."""
        if landmarks is None or not known_faces:
            return None, 0.0

        try:
            # Get face descriptor
            face_descriptor = self.get_face_descriptor(frame, landmarks)
            if face_descriptor is None:
                return None, 0.0

            # Find best matching face
            best_match = None
            best_percentage = 0.0

            # Compare with all known faces
            for name, known_descriptor in known_faces.items():
                percentage = self.compare_faces(face_descriptor, known_descriptor)
                
                if percentage > best_percentage:
                    best_percentage = percentage
                    best_match = name

            return best_match, best_percentage

        except Exception as e:
            logging.error(f"Face verification error: {e}")
            return None, 0.0 