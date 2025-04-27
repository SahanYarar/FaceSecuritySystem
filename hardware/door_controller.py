import time
import logging
from gpiozero import AngularServo # type: ignore (gpiozero type hint sorunları için)
from gpiozero.exc import BadPinFactory
from utils.helpers import handle_error
from common.constants import (
    SERVO_PIN, SERVO_MIN_PULSE, SERVO_MAX_PULSE,
    SERVO_FRAME_WIDTH, CLOSED_ANGLE, OPEN_ANGLE
)

class DoorController:
    """Servo motoru kullanarak kapıyı kontrol eder."""
    def __init__(self):
        self.servo = None
        self.simulated_state = False  # False = closed, True = open
        self.gpio_available = True
        self.last_action_time = 0

        try:
            logging.info("Servo başlatılıyor...")
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
        except BadPinFactory:
            self.gpio_available = False
            logging.warning("GPIO not available, running in simulation mode")
        except Exception as e:
            self.gpio_available = False
            handle_error(f"Servo başlatılamadı (GPIO{SERVO_PIN}): {e}. Servo kontrolü devre dışı.", "Servo Başlatma")
            self.servo = None

    def _move_servo(self, angle):
        """Servo'yu belirtilen açıya hareket ettirir."""
        if not self.gpio_available:
            self.simulated_state = (angle == OPEN_ANGLE)
            return True
        try:
            self.servo.angle = angle
            time.sleep(0.5) # Hareketin tamamlanmasını bekle
            self.servo.detach() # Titreşimi önle
            return True
        except Exception as e:
            return handle_error(f"Servo hareket hatası: {e}", False)

    def open_door(self):
        """Kapıyı açar."""
        if not self.simulated_state:
            logging.info("Kapı açılıyor...")
            if self._move_servo(OPEN_ANGLE):
                self.simulated_state = True
                self.last_action_time = time.time()
                logging.info("Kapı başarıyla açıldı.")
                return True
            else:
                handle_error("Kapı açma işlemi başarısız.", "Kapı Açma")
                return False
        return False # Zaten açıktı

    def close_door(self):
        """Kapıyı kapatır."""
        if self.simulated_state:
            logging.info("Kapı kapatılıyor...")
            if self._move_servo(CLOSED_ANGLE):
                self.simulated_state = False
                self.last_action_time = time.time()
                logging.info("Kapı başarıyla kapatıldı.")
                return True
            else:
                handle_error("Kapı kapatma işlemi başarısız.", "Kapı Kapatma")
                return False
        return False # Zaten kapalıydı

    def get_state(self):
        """Kapının açık olup olmadığını döndürür."""
        if not self.gpio_available:
            return self.simulated_state
        try:
            return self.servo.angle == OPEN_ANGLE
        except Exception as e:
            handle_error(f"Servo durumunu alırken hata oluştu: {e}", False)
            return False

    def cleanup(self):
        """Servo'yu kapatır ve kaynakları serbest bırakır."""
        if self.gpio_available and self.servo:
            try:
                # Kapanış pozisyonuna getir ve kapat
                if self.simulated_state:
                    self.close_door()
                # Servo kaynağını serbest bırak
                self.servo.close()
                logging.info("Servo başarıyla kapatıldı.")
            except Exception as e:
                handle_error(f"Servo kapatılırken hata oluştu: {e}", "Servo Temizleme")
        self.servo = None 