#!/usr/bin/env python3
# -- coding: utf-8 --

import cv2
import time
import logging
import numpy as np
import dlib
from scipy.spatial import distance as dist
from gpiozero import AngularServo # type: ignore (gpiozero type hint sorunları için)
from tinydb import TinyDB        # type: ignore (tinydb type hint sorunları için)
import collections
import traceback # Hata ayıklama için
import math # Euler açıları için

# ———————————————— AYARLAR ————————————————
# Kamera
CAMERA_SRC = 0
FRAME_WIDTH = 640
PROCESS_WIDTH = 320

# Tanıma
RECOG_DIST_THRESH = 0.55
STREAK_THRESHOLD  = 4

# Liveness (Blink + Centroid Move + Pose Variation + Look Left/Right)
EAR_THRESHOLD         = 0.22  # Göz Kırpma Oranı Eşiği
EAR_CONSEC_FRAMES     = 2     # Göz kırpma için ardışık kapalı kare
REQUIRED_BLINKS       = 2     # Gerekli göz kırpma sayısı
LIVENESS_TIMEOUT_FRAMES  = 55 # Canlılık zaman aşımı (kare) (Yeni check için artırıldı)
# -- Kafa Merkezi Hareketi --
HEAD_MOVEMENT_FRAMES  = 10
MIN_CENTROID_MOVEMENT = 0.7
MAX_CENTROID_MOVEMENT = 6.0
CENTROID_LANDMARK_INDICES = [30, 33, 8, 36, 45, 48, 54]
# -- Kafa Pozu Değişimi (Genel Hareketlilik) --
POSE_HISTORY_FRAMES   = 15
MIN_POSE_STD_DEV_SUM  = 0.45  # Genel poz değişimi min eşiği (sadece tamamen hareketsizliği yakalamak için)
POSE_LANDMARK_INDICES = [30, 8, 36, 45, 48, 54]
# -- Sağa/Sola Bakma (YENİ) --
LOOK_LEFT_RIGHT_ANGLE_THRESH = 20.0 # Sağa/Sola bakma için gereken minimum Yaw açısı sapması (derece)

# Kapı / Servo
SERVO_PIN         = 18; SERVO_MIN_PULSE   = 0.0005; SERVO_MAX_PULSE   = 0.0025
SERVO_FRAME_WIDTH = 0.02; CLOSED_ANGLE      = 10; OPEN_ANGLE        = -70
DOOR_OPEN_TIME    = 5.0

# Dosyalar
SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
FACE_REC_MODEL_PATH  = 'dlib_face_recognition_resnet_model_v1.dat'
KNOWN_FACES_DB_PATH  = 'known_faces.json'

# Renkler & UI
COLOR_GREEN=(0,255,0); COLOR_RED=(0,0,255); COLOR_BLUE=(255,0,0); COLOR_YELLOW=(0,255,255);
COLOR_WHITE=(255,255,255); COLOR_BLACK=(0,0,0); COLOR_GRAY=(200,200,200)
UI_BUTTON_HEIGHT=40; UI_BUTTON_WIDTH=120; UI_BUTTON_MARGIN=10
UI_INFO_POS=(10,20); UI_STATUS_POS=(10,45); UI_LIVENESS_POS=(10,70)
UI_INPUT_POS=(10,360); UI_BUTTON_START_Y=400
# ————————————————————————————————————————————

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Yardımcı Fonksiyonlar ---
def eye_aspect_ratio(eye):
    """Verilen göz landmarkları için Göz Açıklık Oranını (EAR) hesaplar."""
    try:
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C) if C > 1e-4 else 0.0
    except Exception as e:
        logging.error(f"EAR hesaplama hatası: {e}")
        return 0.0

def shape_to_np(shape, dtype="int"):
    """Dlib shape nesnesini NumPy array'ine dönüştürür."""
    if shape is None:
        return np.zeros((0, 2), dtype=dtype)
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def calculate_iou(boxA_rect, boxB_rect):
    """İki dlib dikdörtgeni arasındaki Intersection over Union (IoU) değerini hesaplar."""
    if boxA_rect is None or boxB_rect is None:
        return 0.0
    try:
        xA = max(boxA_rect.left(), boxB_rect.left())
        yA = max(boxA_rect.top(), boxB_rect.top())
        xB = min(boxA_rect.right(), boxB_rect.right())
        yB = min(boxA_rect.bottom(), boxB_rect.bottom())
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA_rect.right() - boxA_rect.left()) * (boxA_rect.bottom() - boxA_rect.top())
        boxBArea = (boxB_rect.right() - boxB_rect.left()) * (boxB_rect.bottom() - boxB_rect.top())
        denominator = float(boxAArea + boxBArea - interArea)
        return interArea / denominator if denominator > 0 else 0.0
    except Exception as e:
        logging.error(f"IoU hesaplama hatası: {e}")
        return 0.0

# --- Sınıflar ---
class CameraManager:
    """Kamerayı bulur, başlatır ve ayarlarını yapar."""
    @staticmethod
    def init_camera(width=FRAME_WIDTH, height=None):
        indices_to_try = [0, 1, 2, -1]
        cap = None
        for idx in indices_to_try:
            try:
                cap = cv2.VideoCapture(idx)
                if cap is not None and cap.isOpened():
                    logging.info(f"Kamera {idx} bulundu, ayarlar yapılıyor...")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    # Bazı kameralar FPS ayarını desteklemeyebilir
                    try: cap.set(cv2.CAP_PROP_FPS, 30)
                    except: logging.warning(f"Kamera {idx} için FPS ayarlanamadı.")
                    # Buffer size'ı ayarlama (bazı sistemlerde işe yarayabilir)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # Ayarlar sonrası tekrar kontrol et
                    if cap.isOpened():
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        logging.info(f"Kamera {idx} başarıyla açıldı ({w}x{h}).")
                        return cap # Başarılı, kamerayı döndür
                    else:
                        logging.warning(f"Kamera {idx} ayarlar sonrası açılamadı.")
                        cap.release()
                elif cap:
                    cap.release() # Açılmadıysa serbest bırak
            except Exception as e:
                 logging.error(f"Kamera {idx} denenirken hata: {e}")
                 if cap: cap.release()

        logging.error("Uygun kamera bulunamadı veya açılamadı.")
        logging.error("Kontrol: Kamera bağlı mı? İzinler tamam mı? Doğru indeks mi?")
        return None # Hiçbir kamera bulunamadı

class FaceProcessor:
    """Yüz algılama, landmark, tanıma ve canlılık (EAR, Centroid, Pose) işlemlerini yönetir."""
    def __init__(self):
        try:
            logging.info("Dlib modelleri yükleniyor...")
            self.detector = dlib.get_frontal_face_detector() # type: ignore
            # Model dosyalarının varlığını kontrol etmek iyi bir pratik olabilir
            # import os
            # if not os.path.exists(SHAPE_PREDICTOR_PATH): raise FileNotFoundError(f"Shape predictor not found: {SHAPE_PREDICTOR_PATH}")
            # if not os.path.exists(FACE_REC_MODEL_PATH): raise FileNotFoundError(f"Face recognition model not found: {FACE_REC_MODEL_PATH}")
            self.shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH) # type: ignore
            self.recognizer = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH) # type: ignore
            self.distance_threshold = RECOG_DIST_THRESH
            logging.info(f"Dlib modelleri yüklendi. Mesafe Eşiği: {self.distance_threshold:.4f}")
        except Exception as e:
            logging.error(f"Dlib modelleri yüklenirken kritik hata: {e}")
            logging.error("Model dosyalarının doğru yolda olduğundan emin olun!")
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
            logging.error(f"Yüz algılama hatası: {e}")
            return [] # Hata durumunda boş liste döndür

    def get_landmarks(self, gray_frame, face_rect):
        """Verilen yüz dikdörtgeni için 68 landmark noktasını bulur."""
        try:
            return self.shape_predictor(gray_frame, face_rect)
        except Exception as e:
            logging.error(f"Landmark alınamadı: {e}")
            return None

    def get_face_descriptor(self, frame, landmarks):
        """Verilen yüz ve landmarklar için 128D yüz tanımlayıcı vektörünü hesaplar."""
        if landmarks is None:
            return None
        try:
            # compute_face_descriptor(image, face_landmarks, num_jitters=0, padding=0.25)
            # num_jitters > 0 can improve accuracy but slows down processing.
            return np.array(self.recognizer.compute_face_descriptor(frame, landmarks, 1))
        except Exception as e:
            logging.error(f"Face descriptor alınamadı: {e}")
            return None

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
                # logging.debug(f"Distance to {name}: {d:.4f}") # Debug
                if d < min_dist:
                    min_dist = d
                    best_match = name
            except Exception as e:
                logging.error(f"Identify: '{name}' ile mesafe hesaplama hatası: {e}")

        if best_match and min_dist <= self.distance_threshold:
            confidence = max(0.0, (self.distance_threshold - min_dist) / self.distance_threshold) * 100
            # logging.debug(f"Match found: {best_match} (Dist: {min_dist:.4f}, Conf: {confidence:.1f}%)")
            return best_match, confidence
        else:
            # logging.debug(f"No match found (Min Dist: {min_dist:.4f})")
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
            # logging.debug(f"Centroid Movement Magnitude: {movement_magnitude:.4f}") # Debug

            if MIN_CENTROID_MOVEMENT <= movement_magnitude <= MAX_CENTROID_MOVEMENT:
                return True  # Yeterli hareket var
            elif movement_magnitude < MIN_CENTROID_MOVEMENT:
                 # Yeterli kare biriktiyse ve hala hareket yoksa False döndür
                 return False if len(centroid_history) >= HEAD_MOVEMENT_FRAMES else None
            else: # movement_magnitude > MAX_CENTROID_MOVEMENT
                # logging.debug("Too much centroid movement, maybe unstable tracking?")
                return None # Çok fazla hareket var, kararsız

        except Exception as e:
            logging.error(f"Kafa Hareketi Analiz Hatası: {e}")
            return None

    def get_head_pose_angles(self, shape, frame_size):
        """Verilen landmarklar ve kare boyutu ile kafa pozunu (Euler açıları) hesaplar."""
        if shape is None: return None
        try:
            # Gerekli landmarkların varlığını kontrol et
            max_req_idx = max(POSE_LANDMARK_INDICES)
            if shape.num_parts <= max_req_idx:
                logging.warning("Poz tahmini için yeterli landmark yok.")
                return None

            image_points = np.array([(shape.part(i).x, shape.part(i).y) for i in POSE_LANDMARK_INDICES], dtype=np.float64)
            height, width = frame_size # (H, W)
            focal_length = float(width) # Basit varsayım
            center = (width / 2.0, height / 2.0)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64) # Bozulma yok varsayımı

            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points_3d, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)

            if not success:
                # logging.warning("solvePnP başarısız.") # Çok sık loglamaması için kapatılabilir
                return None

            # Dönüş vektörünü Euler açılarına çevir
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

            # Açıları dereceye çevir (Pitch, Yaw, Roll)
            pitch = math.degrees(x)
            yaw = math.degrees(y)
            roll = math.degrees(z)
            # logging.debug(f"Pose Angles (P,Y,R): ({pitch:.1f}, {yaw:.1f}, {roll:.1f})") # Debug
            return (pitch, yaw, roll)

        except Exception as e:
            logging.error(f"Kafa pozu hesaplama hatası: {e}")
            return None

    def check_liveness_pose_variation(self, pose_history):
        """Poz geçmişine bakarak yeterli AÇISAL DEĞİŞİM olup olmadığını kontrol eder (Tamamen hareketsizliği yakalamak için)."""
        min_req = max(3, POSE_HISTORY_FRAMES // 2)
        if len(pose_history) < min_req:
            return None # Henüz yeterli veri yok
        try:
            history_array = np.array(list(pose_history))
            if history_array.ndim != 2 or history_array.shape[1] != 3: return None
            std_devs = np.std(history_array, axis=0)
            total_std_dev = np.sum(std_devs)
            # logging.debug(f"Pose StdDev Total: {total_std_dev:.4f}") # Debug

            # Yeterli kare biriktiyse ve değişim çok düşükse, fotoğraf şüphesi
            if len(pose_history) >= POSE_HISTORY_FRAMES and total_std_dev < MIN_POSE_STD_DEV_SUM:
                return False # Çok Düşük Açısal Değişim -> FOTOĞRAF ŞÜPHESİ
            # Değişim yeterliyse veya henüz karar vermek için erken ise
            elif total_std_dev >= MIN_POSE_STD_DEV_SUM:
                 return True # Yeterli genel açısal değişim var
            else: # Henüz yeterli kare yok ama hareket az
                 return None # Kararsız

        except Exception as e:
            logging.error(f"Poz Değişimi Analiz Hatası: {e}")
            return None

class FaceStorage:
    """Bilinen yüzlerin descriptorlarını TinyDB kullanarak saklar ve yönetir."""
    def __init__(self, db_path=KNOWN_FACES_DB_PATH):
        self.db_path = db_path
        try:
            self.db = TinyDB(db_path, indent=4, ensure_ascii=False, encoding='utf-8')
        except Exception as e:
            logging.exception(f"TinyDB başlatılırken hata ({db_path}): {e}")
            raise # DB açılamazsa devam etmenin anlamı yok
        self.known_faces = {}
        self.load_known_faces()

    def load_known_faces(self):
        """Veritabanından bilinen yüzleri yükler."""
        self.known_faces = {}
        loaded_count = 0
        invalid_count = 0
        try:
            # Doğrudan tüm kayıtları almayı dene
            all_entries = self.db.all()
        except Exception as e:
             logging.exception(f"Veritabanı okunurken hata ({self.db_path}): {e}")
             all_entries = [] # Hata varsa boş liste ile devam et

        data_to_process = []
        # TinyDB yapısını kontrol et
        if all_entries:
            if isinstance(all_entries, list) and len(all_entries) > 0:
                # Eğer liste içindeki ilk eleman '_default' anahtarlı bir dict ise (eski yapı?)
                if isinstance(all_entries[0], dict) and '_default' in all_entries[0]:
                    default_table = all_entries[0]['_default']
                    if isinstance(default_table, dict):
                        data_to_process = list(default_table.values())
                    else: # Beklenmedik yapı
                        logging.warning(f"DBLoad: Beklenmedik '_default' tablo yapısı: {type(default_table)}")
                # Doğrudan kayıt listesi ise
                elif all(isinstance(item, dict) for item in all_entries):
                    data_to_process = all_entries
                else:
                    logging.warning(f"DBLoad: Tanınmayan veritabanı yapısı: {all_entries}")
            elif isinstance(all_entries, list): # Boş liste ise
                 pass # Sorun yok
            else: # Liste değilse
                 logging.warning(f"DBLoad: Veritabanı içeriği beklenmedik tipte: {type(all_entries)}")


        for record in data_to_process:
            if isinstance(record, dict) and 'name' in record and 'descriptor' in record:
                name = record['name']
                desc_list = record['descriptor']
                if isinstance(desc_list, list) and len(desc_list) == 128:
                    try:
                        np_desc = np.array(desc_list, dtype=np.float64)
                        if not np.isnan(np_desc).any() and not np.isinf(np_desc).any():
                            self.known_faces[name] = np_desc
                            loaded_count += 1
                        else:
                            logging.warning(f"DBLoad: '{name}' NaN/Inf descriptor, atlandı.")
                            invalid_count += 1
                    except Exception as array_err:
                        logging.warning(f"DBLoad: '{name}' descriptor array hatası ({array_err}), atlandı.")
                        invalid_count += 1
                else:
                    logging.warning(f"DBLoad: '{name}' geçersiz descriptor formatı, atlandı.")
                    invalid_count += 1
            else:
                logging.warning(f"DBLoad: Geçersiz kayıt formatı, atlandı: {record}")
                invalid_count += 1

        if loaded_count > 0:
            logging.info(f"{loaded_count} geçerli yüz kaydı yüklendi.")
        if invalid_count > 0:
            logging.warning(f"{invalid_count} geçersiz kayıt atlandı.")
        if not self.known_faces and invalid_count == 0:
            logging.warning("Veritabanı boş veya geçerli kayıt içermiyor.")

    def save_known_faces(self):
        """Hafızadaki bilinen yüzleri veritabanına kaydeder."""
        try:
            entries = [{'name': name, 'descriptor': desc.tolist()} for name, desc in self.known_faces.items()]
            self.db.truncate() # Önce mevcut veriyi temizle
            if entries:
                self.db.insert_multiple(entries) # Sonra yenilerini ekle
            # logging.info(f"{len(entries)} yüz kaydı veritabanına kaydedildi.") # Debug
        except Exception as e:
            logging.exception(f"Veritabanına kaydetme hatası ({self.db_path}): {e}")

    def add_face(self, name, descriptor):
        """Yeni bir yüzü ekler/günceller."""
        name = name.strip()
        if not name: logging.warning("Yüz eklenemedi: İsim boş."); return False
        if descriptor is None or not isinstance(descriptor, np.ndarray) or descriptor.shape != (128,): logging.warning(f"Yüz eklenemedi ('{name}'): Geçersiz descriptor."); return False
        if np.isnan(descriptor).any() or np.isinf(descriptor).any(): logging.warning(f"Yüz eklenemedi ('{name}'): NaN/Inf descriptor."); return False
        self.known_faces[name] = descriptor.astype(np.float64)
        self.save_known_faces()
        logging.info(f"'{name}' yüzü başarıyla eklendi/güncellendi.")
        return True

    def delete_face(self, name):
        """Verilen isimdeki yüzü siler."""
        name = name.strip()
        if name in self.known_faces:
            del self.known_faces[name]
            self.save_known_faces()
            logging.info(f"'{name}' yüzü başarıyla silindi.")
            return True
        logging.warning(f"Silinecek yüz bulunamadı: '{name}'")
        return False

    def get_known_faces(self):
        """Hafızadaki bilinen yüzler sözlüğünü döndürür."""
        return self.known_faces

class DoorController:
    """Servo motoru kullanarak kapıyı kontrol eder."""
    def __init__(self):
        self.servo = None
        self.is_open = False
        self.last_action_time = 0
        try:
            logging.info("Servo başlatılıyor...")
            # Servo Jitter Uyarısı: Eğer jitter sorunu varsa pigpio kullanmayı düşünün.
            # from gpiozero.pins.pigpio import PiGPIOFactory
            # self.servo = AngularServo(..., pin_factory=PiGPIOFactory())
            self.servo = AngularServo(
                SERVO_PIN,
                min_pulse_width=SERVO_MIN_PULSE,
                max_pulse_width=SERVO_MAX_PULSE,
                frame_width=SERVO_FRAME_WIDTH,
                initial_angle=CLOSED_ANGLE
            )
            time.sleep(0.5) # Pozisyona gitmesi için bekle
            self.servo.detach() # Başlangıçta sinyali kes
            logging.info(f"Kapı kontrolcüsü (Servo GPIO{SERVO_PIN}) başlatıldı.")
        except Exception as e:
            logging.error(f"Servo başlatılamadı (GPIO{SERVO_PIN}): {e}. Servo kontrolü devre dışı.")
            self.servo = None

    def _move_servo(self, angle):
        """Servo'yu belirtilen açıya hareket ettirir."""
        if not self.servo:
            logging.warning("Servo işlemi denendi ancak servo kullanılamıyor.")
            return False
        try:
            # self.servo.attach() # Gerekirse? Genellikle gpiozero bunu kendi yönetir.
            self.servo.angle = angle
            time.sleep(0.5) # Hareketin tamamlanmasını bekle
            self.servo.detach() # Titreşimi önle
            return True
        except Exception as e:
            logging.error(f"Servo {angle}° açısına hareket ettirilemedi: {e}")
            return False

    def open_door(self):
        """Kapıyı açar."""
        if not self.is_open:
            logging.info("Kapı açılıyor...")
            if self._move_servo(OPEN_ANGLE):
                self.is_open = True
                self.last_action_time = time.time()
                logging.info("Kapı başarıyla açıldı.")
                return True
            else:
                logging.error("Kapı açma işlemi başarısız.")
                return False
        return False # Zaten açıktı

    def close_door(self):
        """Kapıyı kapatır."""
        if self.is_open:
            logging.info("Kapı kapatılıyor...")
            if self._move_servo(CLOSED_ANGLE):
                self.is_open = False
                self.last_action_time = time.time()
                logging.info("Kapı başarıyla kapatıldı.")
                return True
            else:
                logging.error("Kapı kapatma işlemi başarısız.")
                return False
        return False # Zaten kapalıydı

    def get_state(self):
        """Kapının açık olup olmadığını döndürür."""
        return self.is_open

    def cleanup(self):
        """Servo'yu kapatır ve kaynakları serbest bırakır."""
        if self.servo:
            try:
                # Kapanış pozisyonuna getir ve kapat
                if self.is_open:
                    self.close_door()
                # Servo kaynağını serbest bırak
                self.servo.close()
                logging.info("Servo başarıyla kapatıldı.")
            except Exception as e:
                logging.warning(f"Servo kapatılırken hata oluştu: {e}")
        self.servo = None

class Interface:
    """OpenCV kullanarak kullanıcı arayüzünü çizer ve yönetir."""
    def __init__(self):
        self.buttons = []
        self.selected_name_for_delete = None
        self.message = ""
        self.message_color = COLOR_WHITE
        self.message_time = 0
        self.message_duration = 3.0

    def set_message(self, text, color=COLOR_GREEN, duration=3.0):
        """Ekranda geçici mesaj gösterir."""
        self.message = text
        self.message_color = color
        self.message_time = time.time()
        self.message_duration = duration
        logging.info(f"UI Mesaj: {text}")

    def draw_ui(self, frame, mode, input_text, known_face_names, system_status):
        """Ana UI elemanlarını kare üzerine çizer."""
        self.buttons = [] # Buton listesini her karede sıfırla
        height, width, _ = frame.shape
        current_y = UI_BUTTON_START_Y
        # Ana sistemden gelen eylem yöneticisini al
        action_handler = system_status.get('action_handler', lambda type, val: None)

        # Genel Bilgiler
        cv2.putText(frame, f"Mod: {mode.upper()}", UI_INFO_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1, cv2.LINE_AA)
        status_text = system_status.get("status", "Durum Yok")
        status_color = system_status.get("color", COLOR_WHITE)
        cv2.putText(frame, f"Durum: {status_text}", UI_STATUS_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)

        # Canlılık Durumu (Sadece Normal Modda)
        if mode == 'normal':
            liveness_text = system_status.get("liveness", "N/A")
            liveness_color = system_status.get("liveness_color", COLOR_WHITE)
            cv2.putText(frame, f"Canlilik: {liveness_text}", UI_LIVENESS_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.6, liveness_color, 1, cv2.LINE_AA)

        # Modlara Göre Butonlar ve Diğer Elemanlar
        button_x = UI_BUTTON_MARGIN
        if mode == "normal":
            button_defs = [
                {"label": "Kayit Modu", "action": lambda: action_handler("set_mode", "register")},
                {"label": "Silme Modu", "action": lambda: action_handler("set_mode", "delete")},
                {"label": "Cikis", "action": lambda: action_handler("quit", None)}
            ]
            btn_width_normal = UI_BUTTON_WIDTH + 20 # Biraz daha geniş butonlar
            for b_def in button_defs:
                rect = (button_x, current_y, btn_width_normal, UI_BUTTON_HEIGHT)
                self.buttons.append({"label": b_def["label"], "rect": rect, "action": b_def["action"]})
                button_x += btn_width_normal + UI_BUTTON_MARGIN

        elif mode == "register":
            # İsim giriş alanı
            cv2.putText(frame, f"Isim: {input_text}_", UI_INPUT_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2, cv2.LINE_AA)
            # Butonlar
            btn_width_confirm = int(UI_BUTTON_WIDTH * 1.5)
            btn_width_cancel = UI_BUTTON_WIDTH
            button_defs = [
                {"label": "Onayla Kayit", "width": btn_width_confirm, "action": lambda: action_handler("register_face", input_text)},
                {"label": "Iptal", "width": btn_width_cancel, "action": lambda: action_handler("set_mode", "normal")}
            ]
            for b_def in button_defs:
                rect = (button_x, current_y, b_def["width"], UI_BUTTON_HEIGHT)
                self.buttons.append({"label": b_def["label"], "rect": rect, "action": b_def["action"]})
                button_x += b_def["width"] + UI_BUTTON_MARGIN

        elif mode == "delete":
            # Silinecek isim listesi
            list_y_start = 100
            list_y = list_y_start
            cv2.putText(frame, "Silinecek Ismi Secin:", (UI_BUTTON_MARGIN, list_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)
            max_items_display = 5
            displayed_names = sorted(known_face_names)[:max_items_display]

            for name in displayed_names:
                is_selected = (name == self.selected_name_for_delete)
                button_color = COLOR_YELLOW if is_selected else COLOR_GRAY
                text_color = COLOR_BLACK
                btn_height_small = int(UI_BUTTON_HEIGHT * 0.8)
                btn_width_large = int(UI_BUTTON_WIDTH * 1.8)
                rect = (UI_BUTTON_MARGIN, list_y, btn_width_large, btn_height_small)
                # Lambda'da n=name kullanarak doğru ismi yakala
                button_def = {"label": name, "rect": rect, "action": lambda n=name: setattr(self, "selected_name_for_delete", n)}
                self.buttons.append(button_def)
                # İsmi buton olarak çiz
                x, y, w, h = button_def["rect"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), button_color, -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), text_color, 1)
                cv2.putText(frame, name, (x + 10, y + int(h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
                list_y += int(btn_height_small * 1.1)

            # Seçili ismi göster
            selected_text = f"Secili: {self.selected_name_for_delete}" if self.selected_name_for_delete else "Secili Isim Yok"
            selected_color = COLOR_YELLOW if self.selected_name_for_delete else COLOR_GRAY
            cv2.putText(frame, selected_text, UI_INPUT_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.7, selected_color, 2, cv2.LINE_AA)

            # Silme ve İptal butonları (listenin altına)
            current_y = max(UI_BUTTON_START_Y, list_y + UI_BUTTON_MARGIN)
            btn_width_delete = int(UI_BUTTON_WIDTH * 1.5)
            btn_width_cancel = UI_BUTTON_WIDTH
            button_defs = [
                {"label": "Secileni Sil", "width": btn_width_delete, "action": lambda: action_handler("delete_face", self.selected_name_for_delete)},
                {"label": "Iptal", "width": btn_width_cancel, "action": lambda: action_handler("set_mode", "normal")}
            ]
            for b_def in button_defs:
                rect = (button_x, current_y, b_def["width"], UI_BUTTON_HEIGHT)
                self.buttons.append({"label": b_def["label"], "rect": rect, "action": b_def["action"]})
                button_x += b_def["width"] + UI_BUTTON_MARGIN

        # Oluşturulan butonları çiz (Silme modundaki isim listesi hariç)
        for btn in self.buttons:
            if mode == "delete" and btn["label"] in known_face_names:
                continue # Bunlar zaten yukarıda çizildi

            x, y, w, h = btn["rect"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GRAY, -1) # Arka plan
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BLACK, 1)  # Kenarlık
            # Metni ortala
            text_size, _ = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            text_x = int(x + (w - text_size[0]) / 2)
            text_y = int(y + (h + text_size[1]) / 2)
            cv2.putText(frame, btn["label"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 1, cv2.LINE_AA)

        # Geçici mesajı göster
        if self.message and time.time() < self.message_time + self.message_duration:
            cv2.putText(frame, self.message, (UI_BUTTON_MARGIN, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.message_color, 2, cv2.LINE_AA)
        elif self.message and time.time() >= self.message_time + self.message_duration:
            self.message = "" # Süre dolduysa temizle

    def handle_click(self, x, y):
        """Fare tıklamasının hangi butona denk geldiğini kontrol eder ve eylemi tetikler."""
        # Tıklanan butonu bul
        clicked_action = None
        clicked_label = "N/A"
        for btn in self.buttons:
            bx, by, bw, bh = btn["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                clicked_action = btn.get("action")
                clicked_label = btn.get("label", "İsimsiz Buton")
                break # İlk eşleşen butonda dur

        # Eğer bir buton bulunduysa eylemini çalıştır
        if clicked_action:
            try:
                logging.info(f"Buton tıklandı: {clicked_label}")
                clicked_action()
                return True # Tıklama işlendi
            except Exception as e:
                logging.error(f"Buton eylemi hatası ('{clicked_label}'): {e}")
                traceback.print_exc() # Tam hatayı yazdır
                return False # Hata oluştu
        else:
            # logging.debug(f"Boş alan tıklandı ({x},{y})") # Debug
            return False # Tıklama bir butona denk gelmedi

# --- Ana Sistem Sınıfı ---
class DoorSecuritySystem:
    """Tüm alt sistemleri yöneten ana sınıf."""
    def __init__(self):
        logging.info("Kapi Guvenlik Sistemi v6 (Look L/R) başlatılıyor...")
        self.cap = None
        self.running = False
        self.initialized_successfully = False

        try:
            # Alt sistemleri başlat
            self.camera_manager = CameraManager()
            self.face_processor = FaceProcessor() # Modelleri yükler
            self.storage = FaceStorage()          # Veritabanını yükler
            self.door_controller = DoorController()   # Servo'yu başlatır
            self.interface = Interface()

            # Kamera başlatma
            self.cap = self.camera_manager.init_camera(width=FRAME_WIDTH)
            if not self.cap:
                raise Exception("Kamera başlatılamadı. Sistem devam edemez.")

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
                "status": "Başlatılıyor...",
                "color": COLOR_YELLOW,
                "liveness": "Bekleniyor",
                "liveness_color": COLOR_WHITE,
                "action_handler": self.handle_ui_action
            }

            self.initialized_successfully = True
            logging.info("Sistem başarıyla başlatıldı ve çalışmaya hazır.")

        except Exception as init_err:
            logging.exception(f"Başlatma sırasında kritik hata: {init_err}")
            self.initialized_successfully = False
            self._cleanup() # Hata durumunda temizlik yapmayı dene

    def handle_ui_action(self, action_type, value):
        """Arayüzden gelen eylemleri işler."""
        if action_type == "set_mode":
            if value in ["normal", "register", "delete"]:
                if self.current_mode != value:
                    logging.info(f"Mod değiştiriliyor: {self.current_mode} -> {value}")
                    self.current_mode = value
                    self.input_text = ""
                    self.interface.selected_name_for_delete = None
                    # Mod değişince tanıma ve canlılık durumunu sıfırla
                    self.reset_recognition_and_liveness_state()
                    # Yeni mod için UI durumunu ayarla
                    if value == "normal": self._update_status("Yuz Bekleniyor", COLOR_BLUE)
                    elif value == "register": self._update_status("Kayit: Yuzunuzu gosterin", COLOR_YELLOW)
                    elif value == "delete": self._update_status("Silme: Isim secin", COLOR_YELLOW)
            else:
                logging.warning(f"Geçersiz mod ayarlanmaya çalışıldı: {value}")
        elif action_type == "quit":
            logging.info("Çıkış komutu alındı.")
            self.running = False
        elif action_type == "register_face":
            self.register_face(value) # Kayıt işlemini başlat
        elif action_type == "delete_face":
             if value: # Silinecek isim seçilmiş mi?
                 self.delete_face(value) # Silme işlemini başlat
             else:
                 self.interface.set_message("Silmek icin bir isim secmelisiniz!", COLOR_RED)

    def reset_recognition_and_liveness_state(self, reset_status_text=True):
        """Tanıma, stabilite ve TÜM canlılık değişkenlerini sıfırlar."""
        # Tanıma/Stabilite
        self.candidate_name = None
        self.recognition_streak_count = 0
        self.stable_match_name = None
        self.last_known_rect = None
        self.last_processed_landmarks = None

        # Canlılık (Tümü)
        self.liveness_passed = False
        self.ear_consec_counter = 0
        self.liveness_check_frame_counter = 0 # Bunu sıfırlamak önemli
        self.blinks_detected_count = 0
        self.centroid_history.clear()
        self.head_movement_ok = None
        self.pose_history.clear()
        self.head_pose_variation_ok = None
        self.initial_yaw = None # YENİ
        self.looked_left = False # YENİ
        self.looked_right = False # YENİ

        # UI Durumu
        self.system_status["liveness"] = "Bekleniyor" if self.current_mode == "normal" else "N/A"
        self.system_status["liveness_color"] = COLOR_WHITE if self.current_mode == "normal" else COLOR_GRAY

        # Ana durum metnini de sıfırla (eğer kapı kapalıysa)
        if reset_status_text and self.current_mode == "normal" and not (self.door_controller and self.door_controller.get_state()):
            self._update_status("Yuz Bekleniyor", COLOR_BLUE)
        # logging.debug("Tanıma ve Canlılık durumu sıfırlandı.")

    def _update_status(self, message, color):
        """UI'daki ana durum metnini ve rengini günceller."""
        self.system_status["status"] = message
        self.system_status["color"] = color

    # --- Ana Döngü Adımları ---
    def _process_frame(self, frame):
        """Gelen kareyi alır, işlem boyutuna küçültür ve gri tonlamaya çevirir."""
        try:
            orig_h, orig_w = frame.shape[:2]
            self.scale_factor = PROCESS_WIDTH / orig_w
            process_height = int(orig_h * self.scale_factor)
            # İşlem boyutunu sakla (H, W) - Poz tahmini için lazım
            self.current_process_frame_size = (process_height, PROCESS_WIDTH)
            # Yeniden boyutlandır (INTER_AREA genellikle küçültme için iyidir)
            process_frame = cv2.resize(frame, (PROCESS_WIDTH, process_height), interpolation=cv2.INTER_AREA)
            # Gri tonlamaya çevir
            gray_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
            return process_frame, gray_frame
        except Exception as e:
            logging.error(f"Kare işleme hatası (_process_frame): {e}")
            return None, None

    def _detect_and_process_face(self, gray_frame, process_frame):
        """Gri karede yüz algılar, en büyük yüzü seçer, landmark ve descriptor alır."""
        detected_rects = self.face_processor.detect_faces(gray_frame)
        best_candidate_rect_small = None
        landmarks = None
        face_descriptor = None

        if detected_rects:
            self.last_detection_time = time.time()
            # En büyük yüzü seç
            best_candidate_rect_small = max(detected_rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))

            # Landmarkları al
            landmarks = self.face_processor.get_landmarks(gray_frame, best_candidate_rect_small)
            # Son işlenen landmarkları sakla (Canlılık için önemli)
            self.last_processed_landmarks = landmarks

            if landmarks:
                # Descriptor'ı al
                face_descriptor = self.face_processor.get_face_descriptor(process_frame, landmarks)

                # Orijinal karedeki koordinatları hesapla (çizim ve IoU için)
                current_face_rect_display = dlib.rectangle( # type: ignore
                    int(best_candidate_rect_small.left() / self.scale_factor),
                    int(best_candidate_rect_small.top() / self.scale_factor),
                    int(best_candidate_rect_small.right() / self.scale_factor),
                    int(best_candidate_rect_small.bottom() / self.scale_factor)
                )

                # Takip kontrolü: Önceki bilinen konumla IoU < 0.25 ise durumu sıfırla
                if self.last_known_rect and calculate_iou(self.last_known_rect, current_face_rect_display) < 0.25:
                    logging.info("Yüz konumu çok değişti veya farklı yüz. Durum sıfırlanıyor.")
                    self.reset_recognition_and_liveness_state(reset_status_text=False)

                # Son konumu güncelle
                self.last_known_rect = current_face_rect_display
            else:
                 # Landmark alınamadıysa kutuyu da temizle
                 self.last_known_rect = None
                 logging.warning("Yüz algılandı ancak landmark alınamadı.")
        else:
            # Bir süredir yüz algılanmıyorsa durumu sıfırla
            # (Örn: 1 saniyeden fazla geçtiyse)
            time_since_last_detection = time.time() - self.last_detection_time
            if (self.stable_match_name or self.candidate_name or self.last_known_rect) and time_since_last_detection > 1.0:
                # logging.info("Bir süredir yüz algılanmıyor, durum sıfırlanıyor.") # Çok sık loglamaması için kapatılabilir
                self.reset_recognition_and_liveness_state()
            self.last_known_rect = None # Kutu yoksa None yap

        # Hem descriptor'ı hem de landmarkları döndür
        return face_descriptor, landmarks

    def _update_recognition_stability(self, face_descriptor):
        """Alınan descriptor ile tanıma yapar ve tanıma stabilitesini günceller."""
        is_stable_now = False
        recognized_name = None

        if face_descriptor is not None:
            known_faces_db = self.storage.get_known_faces()
            recognized_name, confidence = self.face_processor.identify_face(face_descriptor, known_faces_db)

        if recognized_name:
            if recognized_name == self.candidate_name:
                self.recognition_streak_count += 1
            else: # Yeni aday veya farklı aday
                # Stabil isim değiştiyse logla ve sıfırla
                if self.stable_match_name and self.stable_match_name != recognized_name:
                    logging.info(f"Stabil isim değişti: {self.stable_match_name} -> {recognized_name}. Canlılık sıfırlanıyor.")
                    self.reset_recognition_and_liveness_state(reset_status_text=False)
                elif not self.candidate_name: # İlk defa bir aday bulunduysa
                     logging.info(f"Yeni aday algilandi: {recognized_name}")

                self.candidate_name = recognized_name
                self.recognition_streak_count = 1 # Sayacı başlat

            # Stabilite eşiğine ulaşıldı mı?
            if self.recognition_streak_count >= STREAK_THRESHOLD:
                # Sadece ilk stabil olduğunda logla ve canlılığı başlat
                if not self.stable_match_name:
                    logging.info(f"Stabiliteye ulaşıldı: {self.candidate_name}")
                    # --- Canlılık kontrolünü BAŞLAT ---
                    self.liveness_check_frame_counter = 1 # Sayacı başlat
                    # Tüm canlılık değişkenlerini sıfırla (önceki denemelerden kalmasın)
                    self.blinks_detected_count = 0
                    self.ear_consec_counter = 0
                    self.centroid_history.clear()
                    self.head_movement_ok = None
                    self.pose_history.clear()
                    self.head_pose_variation_ok = None
                    self.initial_yaw = None # Henüz hesaplanmadı
                    self.looked_left = False
                    self.looked_right = False
                    logging.info(f"'{self.candidate_name}' için canlılık kontrolü başlatılıyor (Gerekli: {REQUIRED_BLINKS} Blink, Kafa Hareketi, Sağa/Sola Bakma).")
                    # --- ---

                self.stable_match_name = self.candidate_name
                is_stable_now = True
            else: # Henüz stabil değil
                self.stable_match_name = None
                is_stable_now = False
                # Eğer canlılık kontrolü başlamışsa (ama stabilite kaybolduysa) durdur
                if self.liveness_check_frame_counter > 0:
                     self.liveness_check_frame_counter = 0
                     # logging.debug("Stabilite kaybedildi, canlılık sayacı sıfırlandı.")
        else: # Yüz tanınmadı veya hiç yüz yok
             if self.stable_match_name: # Eğer önceden stabil bir isim varsa logla
                 logging.info(f"Stabilite kaybedildi (tanıma yok): {self.stable_match_name}")
             # Tanıma olmayınca veya yüz kaybolunca durumu sıfırla
             # (reset_recognition_and_liveness_state içinde zaten sıfırlanıyor)
             is_stable_now = False
             # Eğer yüz yoksa zaten _detect_and_process_face sonunda sıfırlanır
             # Eğer yüz var ama tanınmadıysa, burada sıfırlamaya gerek yok,
             # reset_recognition_and_liveness_state zaten stabilite gidince çağrılır.

        return is_stable_now

    def _update_liveness_status(self, landmarks):
        """Stabil yüz için canlılık kontrolünü (Blink, Centroid, Pose Var, Look L/R) günceller."""
        # Koşullar: Normal mod, Stabil isim var, Canlılık geçilmemiş, Kontrol başlamış
        if not (self.current_mode == "normal" and self.stable_match_name and
                not self.liveness_passed and self.liveness_check_frame_counter > 0):
            return

        # Landmarklar kontrol için gerekli
        if landmarks is None:
             # logging.debug("Canlılık kontrolü için landmark yok, atlanıyor.")
             return # Landmark yoksa bu karede kontrol yapma

        # Zaman aşımı kontrolü
        if self.liveness_check_frame_counter > LIVENESS_TIMEOUT_FRAMES:
            logging.warning(f"Canlılık ZAMAN AŞIMI: {self.stable_match_name}")
            self.interface.set_message(f"{self.stable_match_name}: Canlilik Zaman Asimi", COLOR_RED)
            self.reset_recognition_and_liveness_state()
            return

        # --- Canlılık Kontrollerini Güncelle ---

        # 1. Göz Kırpma (EAR)
        ear_value = self.face_processor.check_liveness_ear(landmarks)
        if ear_value is not None:
            if ear_value < EAR_THRESHOLD: self.ear_consec_counter += 1
            else:
                if self.ear_consec_counter >= EAR_CONSEC_FRAMES: self.blinks_detected_count += 1
                self.ear_consec_counter = 0

        # 2. Kafa Merkezi Hareketi (Centroid)
        centroid = self.face_processor.get_face_centroid(landmarks)
        if centroid is not None:
            self.centroid_history.append(centroid)
            # Sadece None değilse güncelle
            hm_check_result = self.face_processor.check_liveness_head_movement(self.centroid_history)
            if hm_check_result is not None: self.head_movement_ok = hm_check_result

        # 3. Kafa Pozu ve Sağa/Sola Bakma
        pose_angles = self.face_processor.get_head_pose_angles(landmarks, self.current_process_frame_size)
        current_yaw = None
        if pose_angles is not None:
            pitch, yaw, roll = pose_angles
            current_yaw = yaw # Sağa/sola bakma için Yaw açısını al
            self.pose_history.append(pose_angles)
            # Genel poz değişimi durumunu güncelle
            pv_check_result = self.face_processor.check_liveness_pose_variation(self.pose_history)
            if pv_check_result is not None: self.head_pose_variation_ok = pv_check_result

            # Başlangıç Yaw açısını kaydet (eğer henüz kaydedilmediyse)
            if self.initial_yaw is None:
                self.initial_yaw = current_yaw
                logging.info(f"Canlılık: Başlangıç Yaw açısı {self.initial_yaw:.1f} derece olarak ayarlandı.")

            # Sağa/Sola bakma kontrolü (sadece initial_yaw ayarlandıktan sonra)
            if self.initial_yaw is not None and current_yaw is not None:
                # Sola bakıldı mı?
                if not self.looked_left and current_yaw < self.initial_yaw - LOOK_LEFT_RIGHT_ANGLE_THRESH:
                    self.looked_left = True
                    logging.info(f"Canlılık: Sola bakma algılandı (Yaw: {current_yaw:.1f})")
                # Sağa bakıldı mı?
                if not self.looked_right and current_yaw > self.initial_yaw + LOOK_LEFT_RIGHT_ANGLE_THRESH:
                    self.looked_right = True
                    logging.info(f"Canlılık: Sağa bakma algılandı (Yaw: {current_yaw:.1f})")

        # --- Karar Verme ---
        # Anlık durumları al
        blinks_ok = self.blinks_detected_count >= REQUIRED_BLINKS
        head_move_ok = self.head_movement_ok is True
        pose_vary_ok = self.head_pose_variation_ok is True # Genel hareketlilik
        look_lr_ok = self.looked_left and self.looked_right # Sağa ve sola bakıldı mı?

        # --- Başarısızlık Koşulları (Öncelikli) ---
        # 1. Zaman Aşımı (Yukarıda kontrol edildi)
        # 2. Kesin Centroid Hareketsizliği
        if self.head_movement_ok is False:
            logging.warning(f"Canlılık BAŞARISIZ: {self.stable_match_name} (Centroid Hareketsiz!)")
            self.interface.set_message(f"{self.stable_match_name}: Canlilik Basarisiz (Hareketsiz)", COLOR_RED)
            self.reset_recognition_and_liveness_state(); return
        # 3. Kesin Poz Hareketsizliği (Genel)
        if self.head_pose_variation_ok is False:
            logging.warning(f"Canlılık BAŞARISIZ: {self.stable_match_name} (Poz Hareketsiz - FOTOĞRAF?)")
            self.interface.set_message(f"{self.stable_match_name}: Canlilik Basarisiz (Fotograf?)", COLOR_RED)
            self.reset_recognition_and_liveness_state(); return
        # Not: Hareket/Poz tutarsızlığı kontrolü kaldırıldı, L/R kontrolü daha güçlü.

        # --- Başarı Koşulu ---
        # Blink + Hareket + Sağa/Sola Bakma Tamamlandı mı?
        if blinks_ok and head_move_ok and look_lr_ok:
            self.liveness_passed = True
            logging.info(f"Canlılık kontrolü BAŞARILI: {self.stable_match_name} (B:{blinks_ok}, HM:{head_move_ok}, L/R:{look_lr_ok})")
            self.system_status["liveness"] = "Gecildi"
            self.system_status["liveness_color"] = COLOR_GREEN
            return # Başarılı, fonksiyondan çık

        # --- Devam Ediyor Durumu ---
        else:
            # Detaylı durum gösterimi
            status_parts = []
            status_parts.append(f"B:{self.blinks_detected_count}/{REQUIRED_BLINKS}")
            hm_status = "?" if self.head_movement_ok is None else ("OK" if head_move_ok else "YOK")
            status_parts.append(f"HM:{hm_status}")
            # pv_status = "?" if self.head_pose_variation_ok is None else ("OK" if pose_vary_ok else "YOK")
            # status_parts.append(f"PV:{pv_status}") # Genel poz durumunu göstermeyebiliriz artık
            lr_status = ""
            if self.looked_left and self.looked_right: lr_status = "OK"
            elif self.looked_left: lr_status = "L"
            elif self.looked_right: lr_status = "R"
            else: lr_status = "N" # None
            status_parts.append(f"L/R:{lr_status}") # Sağa/Sola bakma durumu
            status_parts.append(f"{self.liveness_check_frame_counter}/{LIVENESS_TIMEOUT_FRAMES}F")
            self.system_status["liveness"] = f"Kontrol ({', '.join(status_parts)})"
            self.system_status["liveness_color"] = COLOR_YELLOW

        # Canlılık kontrolü sayacını artır (her adımdan sonra, timeout kontrolünden önce)
        self.liveness_check_frame_counter += 1


    def _update_door_state(self, is_stable_now):
        """Tanıma ve canlılık durumuna göre kapıyı açar veya kapatır."""
        # Kapıyı Açma Koşulu
        if (self.current_mode == "normal" and is_stable_now and
                self.liveness_passed and self.door_controller and not self.door_controller.get_state()):
            logging.info(f"Kapı açma koşulları sağlandı: {self.stable_match_name}")
            if self.door_controller.open_door():
                self.door_opened_time = time.time()
                self._update_status(f"Kapi Acildi ({self.stable_match_name})", COLOR_GREEN)
            else:
                 # Kapı açılamadıysa durumu güncelle
                 self._update_status("Kapi Acma Hatasi!", COLOR_RED)

        # Kapıyı Kapatma Koşulu
        if (self.door_controller and self.door_controller.get_state() and
                self.door_opened_time and (time.time() - self.door_opened_time > DOOR_OPEN_TIME)):
            logging.info("Kapı açık kalma süresi doldu, kapatılıyor.")
            if self.door_controller.close_door():
                self.door_opened_time = None # Kapı kapandı, zamanı sıfırla
                self._update_status("Kapi Kapatildi", COLOR_RED)
                # Kapı kapandıktan sonra tanıma ve canlılık durumunu sıfırla
                self.reset_recognition_and_liveness_state(reset_status_text=True)
            else:
                 # Kapı kapatılamadıysa durumu güncelle
                 self._update_status("Kapi Kapatma Hatasi!", COLOR_RED)
                 logging.error("Kapı kapatılamadı!")

    def _update_ui_status_text(self, is_stable_now):
        """Mevcut duruma göre UI'daki ana durum metnini günceller."""
        # Sadece normal modda ve kapı mesajı yokken güncelle
        if self.current_mode == "normal" and "Kapi Ac" not in self.system_status["status"] and "Kapi Kapat" not in self.system_status["status"]:
            if self.stable_match_name:
                if self.liveness_passed:
                    self._update_status(f"Tanindi ({self.stable_match_name})", COLOR_GREEN)
                elif self.liveness_check_frame_counter > 0:
                    # Canlılık kontrolü devam ediyor (mesaj _update_liveness_status içinde ayarlanıyor)
                    self._update_status(f"Canlilik Kontrol ({self.stable_match_name})", COLOR_YELLOW)
                else: # Stabil ama canlılık henüz başlamadı
                    self._update_status(f"Tanindi ({self.stable_match_name}-Stabil)", COLOR_GREEN)
            elif self.candidate_name: # Sadece aday var
                self._update_status(f"Taniniyor ({self.candidate_name}-{self.recognition_streak_count}/{STREAK_THRESHOLD})", COLOR_YELLOW)
            elif self.last_known_rect: # Yüz var ama tanınmıyor
                self._update_status("Bilinmeyen Yuz", COLOR_RED)
            else: # Hiç yüz yok
                self._update_status("Yuz Bekleniyor", COLOR_BLUE)

    def _draw_frame_elements(self, display_frame, is_stable_now):
        """Tespit edilen yüzün etrafına kutu ve UI elemanlarını çizer."""
        # Yüz kutusu ve ismi
        if self.last_known_rect:
            x1, y1, x2, y2 = self.last_known_rect.left(), self.last_known_rect.top(), self.last_known_rect.right(), self.last_known_rect.bottom()
            label = "Bilinmeyen"
            color = COLOR_RED

            if self.stable_match_name:
                label = self.stable_match_name
                color = COLOR_GREEN if self.liveness_passed else COLOR_YELLOW
            elif self.candidate_name:
                 label = f"{self.candidate_name} ({self.recognition_streak_count}/{STREAK_THRESHOLD})"
                 color = COLOR_YELLOW

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            font_scale = 0.6; thickness = 1
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_y = y1 - 10 if y1 - 10 > h else y1 + h + 10
            # Etiket arka planı (isteğe bağlı)
            # cv2.rectangle(display_frame, (x1, text_y - h - 2), (x1 + w, text_y + 2), color, -1)
            cv2.putText(display_frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        # Genel UI elemanları
        try:
            known_names = list(self.storage.get_known_faces().keys())
            self.interface.draw_ui(display_frame, self.current_mode, self.input_text, known_names, self.system_status)
        except Exception as e:
            logging.error(f"UI çizim hatası: {e}")
            traceback.print_exc()

    def run(self):
        """Ana uygulama döngüsünü başlatır ve yönetir."""
        if not self.initialized_successfully:
            logging.error("Sistem başlatılamadığı için çalıştırılamıyor.")
            return
        if not self.cap or not self.cap.isOpened():
             logging.error("Kamera mevcut değil veya açılamadı. Çalıştırılamıyor.")
             return

        window_name = "Yuz Tanima Kapi Sistemi v6 (Look L/R)"
        try:
             cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
             cv2.setMouseCallback(window_name, self.handle_mouse_event)
        except Exception as e:
             logging.error(f"OpenCV penceresi oluşturulamadı: {e}. GUI ortamı mevcut mu?")
             return # Pencere yoksa devam etmenin anlamı yok


        self.running = True
        logging.info("Ana döngü başlatıldı. Çıkmak için 'q' tuşuna basın.")
        self.handle_ui_action("set_mode", "normal") # Başlangıç modunu ayarla

        while self.running:
            try:
                # 1. Kareyi al
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logging.warning("Kameradan frame alınamadı, bekleniyor...")
                    time.sleep(0.1)
                    continue

                display_frame = frame.copy() # Çizim için kopya
                self.frame_counter += 1

                # 2. Kareyi işle (küçült, griye çevir)
                process_frame, gray_frame = self._process_frame(frame)
                if process_frame is None or gray_frame is None:
                    continue # Kare işlenemezse atla

                # 3. Yüzü algıla, descriptor ve landmarkları al
                face_descriptor, landmarks = self._detect_and_process_face(gray_frame, process_frame)

                # 4. Normal mod işlemleri (Tanıma, Canlılık, Kapı)
                is_stable_now = False
                if self.current_mode == "normal":
                    is_stable_now = self._update_recognition_stability(face_descriptor)
                    # Canlılık kontrolü için landmarkları kullan
                    self._update_liveness_status(landmarks)
                    self._update_door_state(is_stable_now)

                # 5. UI Durum Metnini Güncelle
                self._update_ui_status_text(is_stable_now)

                # 6. Çizimleri yap
                self._draw_frame_elements(display_frame, is_stable_now)

                # 7. Kareyi Göster
                cv2.imshow(window_name, display_frame)

                # 8. Klavye Girişlerini Kontrol Et
                key = cv2.waitKey(1) & 0xFF

                # 'q' tuşu her zaman çıkış veya normal moda dönüş
                if key == ord('q'):
                    if self.current_mode == "normal":
                        logging.info("'q' tuşuna basıldı, çıkılıyor.")
                        self.running = False
                    else:
                        logging.info("'q' tuşuna basıldı, normal moda dönülüyor.")
                        self.handle_ui_action("set_mode", "normal")

                # Modlara özel klavye kısayolları
                elif self.current_mode == "normal":
                    if key == ord('r'): self.handle_ui_action("set_mode", "register")
                    elif key == ord('d'): self.handle_ui_action("set_mode", "delete")

                # Kayıt veya Silme modunda metin girişi
                elif self.current_mode in ["register", "delete"]:
                    self.handle_text_input(key)

            except KeyboardInterrupt:
                logging.info("CTRL+C algılandı, çıkılıyor.")
                self.running = False
            except Exception as loop_err:
                 logging.exception(f"Ana döngüde beklenmedik hata: {loop_err}")
                 # İsteğe bağlı: Hata durumunda döngüyü durdurabiliriz
                 # self.running = False
                 time.sleep(1) # Hata sonrası kısa bekleme

        self._cleanup() # Döngü bittiğinde temizlik yap

    def handle_mouse_event(self, event, x, y, flags, param):
        """Fare olaylarını (sol tıklama) işler ve arayüze iletir."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.interface.handle_click(x, y)

    def handle_text_input(self, key):
        """Kayıt ve Silme modlarında klavye girişlerini işler."""
        if key == 27: # ESC
            self.handle_ui_action("set_mode", "normal")
        elif key == 13: # Enter
            if self.current_mode == "register":
                self.register_face(self.input_text)
        elif key in [8, 127]: # Backspace / Delete
            self.input_text = self.input_text[:-1]
        elif 32 <= key <= 126: # Yazdırılabilir karakterler
            if len(self.input_text) < 20:
                self.input_text += chr(key)
            else:
                self.interface.set_message("Maksimum isim uzunlugu!", COLOR_YELLOW, duration=2.0)

    def register_face(self, name_to_register):
        """Mevcut karedeki yüzü alır ve verilen isimle kaydeder."""
        if self.current_mode != "register":
            logging.warning("Yüz kaydı sadece 'register' modunda yapılabilir.")
            return

        name = name_to_register.strip()
        if not name:
            self.interface.set_message("Kayit icin gecerli bir isim girin!", COLOR_RED)
            return

        # Anlık kareyi al ve işle
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.interface.set_message("Kayit: Kamera karesi alınamadı!", COLOR_RED); return
        process_frame, gray_frame = self._process_frame(frame)
        if process_frame is None or gray_frame is None:
            self.interface.set_message("Kayit: Kare işlenemedi!", COLOR_RED); return

        # Yüzü bul
        faces = self.face_processor.detect_faces(gray_frame)
        if not faces: self.interface.set_message("Kayit: Yuz bulunamadi!", COLOR_RED); return
        if len(faces) > 1: self.interface.set_message("Kayit: Sadece bir yuz olmali!", COLOR_RED); return

        # Landmark ve descriptor al
        landmarks = self.face_processor.get_landmarks(gray_frame, faces[0])
        if not landmarks: self.interface.set_message("Kayit: Landmark alinamadi!", COLOR_RED); return
        descriptor = self.face_processor.get_face_descriptor(process_frame, landmarks)
        if descriptor is None: self.interface.set_message("Kayit: Descriptor hesaplanamadi!", COLOR_RED); return

        # Kaydet
        if self.storage.add_face(name, descriptor):
            self.interface.set_message(f"'{name}' basariyla kaydedildi!", COLOR_GREEN, duration=5.0)
            self.handle_ui_action("set_mode", "normal") # Normal moda dön
        else:
            self.interface.set_message(f"'{name}' kaydedilirken hata olustu!", COLOR_RED)

    def delete_face(self, name_to_delete):
        """Verilen isimdeki yüz kaydını siler."""
        if self.current_mode != "delete":
            logging.warning("Yüz silme sadece 'delete' modunda yapılabilir.")
            return
        if not name_to_delete:
            self.interface.set_message("Silmek icin bir isim secmelisiniz!", COLOR_RED); return

        if self.storage.delete_face(name_to_delete):
            self.interface.set_message(f"'{name_to_delete}' basariyla silindi.", COLOR_GREEN)
            self.interface.selected_name_for_delete = None # Seçimi temizle
            # Silme modunda kal, durumu güncelle
            self._update_status("Silme: Isim secin veya Iptal", COLOR_YELLOW)
        else:
            # Silme hatası (isim bulunamadı vb.)
            self.interface.set_message(f"'{name_to_delete}' silinirken hata olustu!", COLOR_RED)

    def _cleanup(self):
        """Uygulama kapanırken kaynakları serbest bırakır."""
        logging.info("Temizleme işlemleri başlatılıyor...")
        self.running = False
        time.sleep(0.1) # Diğer işlemlerin bitmesi için kısa bekleme

        # Kamerayı serbest bırak
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            try: self.cap.release(); logging.info("Kamera serbest bırakıldı.")
            except Exception as e: logging.warning(f"Kamera kapatma hatası: {e}")
        self.cap = None

        # OpenCV pencerelerini kapat
        try: cv2.destroyAllWindows(); logging.info("OpenCV pencereleri kapatıldı.")
        except Exception as e: logging.warning(f"OpenCV pencere kapatma hatası: {e}")

        # Kapı kontrolcüsünü temizle
        if hasattr(self, 'door_controller') and self.door_controller:
             try: self.door_controller.cleanup()
             except Exception as e: logging.warning(f"Kapı kontrolcü temizleme hatası: {e}")
        self.door_controller = None

        logging.info("Temizleme tamamlandı.")

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
             # Başlatma başarısız olduysa, finally bloğu zaten cleanup çağıracak.

    except KeyboardInterrupt:
        logging.info("CTRL+C ile program sonlandırıldı.")
        # finally bloğu cleanup'ı halleder.
    except Exception as main_err:
        # Ana kapsamda beklenmedik bir hata olursa logla
        logging.exception(f"Programın ana kısmında beklenmedik ve yakalanmayan hata: {main_err}")
        # finally bloğu cleanup'ı halleder.
    finally:
        # Program sonlanırken (normal veya hata ile) temizliği çağır
        if system:
            logging.info("Program sonlanıyor, temizlik yapılıyor...")
            system._cleanup()
        else:
             logging.info("Sistem nesnesi oluşturulamadığı için temizlik atlandı.")
        logging.info("Program sonlandı.")