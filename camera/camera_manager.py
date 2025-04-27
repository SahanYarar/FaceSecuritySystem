import cv2
import logging
from utils.helpers import handle_error

class CameraManager:
    """Kamerayı bulur, başlatır ve ayarlarını yapar."""
    @staticmethod
    def init_camera(width=640, height=None):
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
                 handle_error(f"Kamera {idx} denenirken hata: {e}", "Kamera Başlatma")
                 if cap: cap.release()

        handle_error("Uygun kamera bulunamadı veya açılamadı.", "Kamera Başlatma")
        handle_error("Kontrol: Kamera bağlı mı? İzinler tamam mı? Doğru indeks mi?", "Kamera Başlatma")
        return None # Hiçbir kamera bulunamadı 