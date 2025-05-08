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
        # Initialize all attributes first
        self.gpio_available = True
        self.servo = None
        self.is_open = False
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
            time.sleep(0.5)  # Pozisyona gitmesi için bekle
            self.servo.detach()  # Başlangıçta sinyali kes
            logging.info(f"Kapı kontrolcüsü (Servo GPIO{SERVO_PIN}) başlatıldı.")
        except Exception as e:
            logging.error(f"Servo başlatma hatası: {e}")
            self.gpio_available = False
            self.servo = None
            handle_error(f"Servo başlatılamadı (GPIO{SERVO_PIN}): {e}. Servo kontrolü devre dışı.", "Servo Başlatma")

    def _move_servo(self, angle):
        """Servo'yu belirtilen açıya hareket ettirir."""
        if not self.gpio_available:
            self.is_open = (angle == OPEN_ANGLE)
            return True
        try:
            # Ensure servo is powered before moving
            if self.servo is None:
                self.servo = AngularServo(
                    SERVO_PIN,
                    min_pulse_width=SERVO_MIN_PULSE,
                    max_pulse_width=SERVO_MAX_PULSE,
                    frame_width=SERVO_FRAME_WIDTH,
                    initial_angle=angle
                )
                time.sleep(0.5)
            
            # Move to desired angle
            logging.info(f"Servo {angle}° açısına hareket ettiriliyor...")
            self.servo.angle = angle
            time.sleep(0.5)  # Hareketin tamamlanması için bekle
            
            # Update state
            self.is_open = (angle == OPEN_ANGLE)
            
            # Detach to prevent vibration
            self.servo.detach()
            logging.info(f"Servo {angle}° açısına başarıyla hareket etti ve detach edildi.")
            return True
        except Exception as e:
            logging.error(f"Servo hareket hatası: {e}")
            return handle_error(f"Servo hareket hatası: {e}", False)

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
                handle_error("Kapı açma işlemi başarısız.", "Kapı Açma")
                return False
        logging.debug("Kapı zaten açık.")
        return False  # Zaten açıktı

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
                handle_error("Kapı kapatma işlemi başarısız.", "Kapı Kapatma")
                return False
        logging.debug("Kapı zaten kapalı.")
        return False  # Zaten kapalıydı

    def get_state(self):
        """Kapının açık olup olmadığını döndürür."""
        return self.is_open

    def cleanup(self):
        """Servo'yu kapatır ve kaynakları serbest bırakır."""
        if self.gpio_available and self.servo:
            try:
                # Kapanış pozisyonuna getir ve kapat
                if self.is_open:
                    self.close_door()
                # Servo kaynağını serbest bırak
                self.servo.close()
                logging.info("Servo başarıyla kapatıldı.")
            except Exception as e:
                handle_error(f"Servo kapatılırken hata oluştu: {e}", "Servo Temizleme")
        self.servo = None 