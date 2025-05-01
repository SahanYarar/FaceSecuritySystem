import time
import logging
from gpiozero import AngularServo # type: ignore (gpiozero type hint sorunları için)
from utils.helpers import handle_error
from common.constants import (
    SERVO_PIN, SERVO_MIN_PULSE, SERVO_MAX_PULSE,
    SERVO_FRAME_WIDTH, CLOSED_ANGLE, OPEN_ANGLE
)

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