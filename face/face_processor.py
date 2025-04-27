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
    POSE_LANDMARK_INDICES, LOOK_LEFT_RIGHT_ANGLE_THRESH
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

    def check_liveness_ear(self, shape):
        """Landmarklardan Göz Açıklık Oranını (EAR) hesaplar."""
        if shape is None: return None
        try:
            coords = shape_to_np(shape)
            max_idx = max(self.l_end, self.r_end) - 1
            if coords.shape[0] <= max_idx:
                logging.warning("EAR için yeterli landmark yok.")
                return None
            left_eye = coords[self.l_start:self.l_end]
            right_eye = coords[self.r_start:self.r_end]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            return ear
        except Exception as e:
            logging.error(f"EAR Hesaplama Hatası: {e}")
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

    def get_head_pose_angles(self, shape, frame_size):
        """Verilen landmarklar ve kare boyutu ile kafa pozunu (Euler açıları) hesaplar."""
        if shape is None: return None
        try:
            max_req_idx = max(POSE_LANDMARK_INDICES)
            if shape.num_parts <= max_req_idx:
                logging.warning("Poz tahmini için yeterli landmark yok.")
                return None

            image_points = np.array([(shape.part(i).x, shape.part(i).y) for i in POSE_LANDMARK_INDICES], dtype=np.float64)
            height, width = frame_size
            focal_length = float(width)
            center = (width / 2.0, height / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points_3d, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)

            if not success:
                return None

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            sy = math.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
            singular = sy < 1e-6

            if not singular:
                x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y = math.atan2(-rotation_matrix[2, 0], sy)
                z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y = math.atan2(-rotation_matrix[2, 0], sy)
                z = 0

            pitch = math.degrees(x)
            yaw = math.degrees(y)
            roll = math.degrees(z)
            return (pitch, yaw, roll)

        except Exception as e:
            logging.error(f"Kafa pozu hesaplama hatası: {e}")
            return None

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
        """İki yüz tanımlayıcısı arasındaki mesafeyi hesaplar."""
        if face_descriptor1 is None or face_descriptor2 is None:
            return float('inf')
        try:
            return np.linalg.norm(face_descriptor1 - face_descriptor2)
        except Exception as e:
            logging.error(f"Yüz karşılaştırma hatası: {e}")
            return float('inf') 