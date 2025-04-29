import cv2
import time
import logging
from common.constants import (
    COLOR_GREEN, COLOR_RED, COLOR_BLUE, COLOR_YELLOW,
    COLOR_WHITE, COLOR_BLACK, COLOR_GRAY,
    UI_BUTTON_HEIGHT, UI_BUTTON_WIDTH, UI_BUTTON_MARGIN,
    UI_INFO_POS, UI_STATUS_POS, UI_LIVENESS_POS,
    UI_INPUT_POS, UI_BUTTON_START_Y
)

class Interface:
    """OpenCV kullanarak kullanıcı arayüzünü çizer ve yönetir."""
    def __init__(self):
        self.buttons = []
        self.selected_name_for_delete = None
        self.message = ""
        self.message_color = COLOR_WHITE
        self.message_time = 0
        self.message_duration = 3.0
        self.status = "Starting..."
        self.status_color = COLOR_BLUE
        self.liveness = "Waiting"
        self.liveness_color = COLOR_WHITE

    def update_status(self, status, color):
        """Update the system status."""
        self.status = status
        self.status_color = color
       # logging.info(f"Status updated: {status}")

    def update_liveness(self, liveness_data):
        """Update the liveness status display."""
        if not liveness_data:
            self.liveness = "Waiting for Face Recognition"
            self.liveness_color = COLOR_WHITE
            return

        status = liveness_data.get("status")
        name = liveness_data.get("name", "Unknown")

        if status == "not_checking":
            self.liveness = "Waiting for Face Recognition"
            self.liveness_color = COLOR_WHITE
        elif status == "timeout":
            self.liveness = f"{name}: Timeout"
            self.liveness_color = COLOR_RED
        elif status == "insufficient_head_movement":
            self.liveness = f"{name}: Insufficient head movement!"
            self.liveness_color = COLOR_RED
        elif status == "low_pose_variation":
            self.liveness = f"{name}: Low pose variation - PHOTO"
            self.liveness_color = COLOR_RED
        elif status == "passed":
            self.liveness = f"{name}: Liveness check PASSED"
            self.liveness_color = COLOR_GREEN
        elif status == "in_progress":
            # Create detailed status display
            status_parts = []
            status_parts.append(f"Blinks: {liveness_data['blinks']}/{liveness_data['required_blinks']}")
            
            # Head movement status
            hm_status = "Waiting" if liveness_data['head_movement'] is None else ("OK" if liveness_data['head_movement'] else "Insufficient")
            status_parts.append(f"Head Movement: {hm_status}")
            
            # Look left/right status
            left_status = "OK" if liveness_data['looked_left'] else "Look left"
            right_status = "OK" if liveness_data['looked_right'] else "Look right"
            status_parts.append(f"Left Look: {left_status}")
            status_parts.append(f"Right Look: {right_status}")
            
            # Timer status
            status_parts.append(f"Time Remaining: {liveness_data['frames_remaining']} frames")

            self.liveness = f"{name}: Liveness Check\n" + "\n".join(status_parts)
            self.liveness_color = COLOR_YELLOW

    def set_message(self, text, color=COLOR_GREEN, duration=3.0):
        """Ekranda geçici mesaj gösterir."""
        self.message = text
        self.message_color = color
        self.message_time = time.time()
        self.message_duration = duration
       # logging.info(f"UI Mesaj: {text}")

    def draw_ui(self, frame, mode, input_text, known_face_names, system_status):
        """Ana UI elemanlarını kare üzerine çizer."""
        self.buttons = [] # Buton listesini her karede sıfırla
        height, width, _ = frame.shape
        current_y = UI_BUTTON_START_Y
        action_handler = system_status.get('action_handler', lambda type, val: None)

        # Genel Bilgiler
        cv2.putText(frame, f"Mod: {mode.upper()}", UI_INFO_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 1, cv2.LINE_AA)
        status_text = self.status
        status_color = self.status_color
        cv2.putText(frame, f"Durum: {status_text}", UI_STATUS_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)

        # Canlılık Durumu (Sadece Normal Modda)
        if mode == 'normal':
            # Split the liveness text into lines and draw each line
            lines = self.liveness.split('\n')
            y_offset = UI_LIVENESS_POS[1]
            for line in lines:
                cv2.putText(frame, line, (UI_LIVENESS_POS[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.liveness_color, 2, cv2.LINE_AA)
                y_offset += 25  # Add some space between lines

            # Draw door remaining time if available
            if system_status.get("door_remaining_time") is not None:
                remaining_time = system_status["door_remaining_time"]
                time_text = f"Door closes in: {remaining_time:.1f}s"
                time_pos = (10, y_offset + 10)  # Position below the liveness status
                cv2.putText(frame, time_text, time_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2, cv2.LINE_AA)

        # Modlara Göre Butonlar ve Diğer Elemanlar
        button_x = UI_BUTTON_MARGIN
        if mode == "normal":
            button_defs = [
                {"label": "Kayit Modu", "action": lambda: action_handler("set_mode", "register")},
                {"label": "Silme Modu", "action": lambda: action_handler("set_mode", "delete")},
                {"label": "Cikis", "action": lambda: action_handler("quit", None)}
            ]
            btn_width_normal = UI_BUTTON_WIDTH + 20
            for b_def in button_defs:
                rect = (button_x, current_y, btn_width_normal, UI_BUTTON_HEIGHT)
                self.buttons.append({"label": b_def["label"], "rect": rect, "action": b_def["action"]})
                button_x += btn_width_normal + UI_BUTTON_MARGIN

        elif mode == "register":
            cv2.putText(frame, f"Isim: {input_text}_", UI_INPUT_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2, cv2.LINE_AA)
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
                button_def = {"label": name, "rect": rect, "action": lambda n=name: setattr(self, "selected_name_for_delete", n)}
                self.buttons.append(button_def)
                x, y, w, h = button_def["rect"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), button_color, -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), text_color, 1)
                cv2.putText(frame, name, (x + 10, y + int(h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
                list_y += int(btn_height_small * 1.1)

            selected_text = f"Secili: {self.selected_name_for_delete}" if self.selected_name_for_delete else "Secili Isim Yok"
            selected_color = COLOR_YELLOW if self.selected_name_for_delete else COLOR_GRAY
            cv2.putText(frame, selected_text, UI_INPUT_POS, cv2.FONT_HERSHEY_SIMPLEX, 0.7, selected_color, 2, cv2.LINE_AA)

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
                continue

            x, y, w, h = btn["rect"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GRAY, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BLACK, 1)
            text_size, _ = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            text_x = int(x + (w - text_size[0]) / 2)
            text_y = int(y + (h + text_size[1]) / 2)
            cv2.putText(frame, btn["label"], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 1, cv2.LINE_AA)

        # Geçici mesajı göster
        if self.message and time.time() < self.message_time + self.message_duration:
            cv2.putText(frame, self.message, (UI_BUTTON_MARGIN, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.message_color, 2, cv2.LINE_AA)
        elif self.message and time.time() >= self.message_time + self.message_duration:
            self.message = ""

    def handle_click(self, x, y):
        """Fare tıklamasının hangi butona denk geldiğini kontrol eder ve eylemi tetikler."""
        clicked_action = None
        clicked_label = "N/A"
        for btn in self.buttons:
            bx, by, bw, bh = btn["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                clicked_action = btn.get("action")
                clicked_label = btn.get("label", "İsimsiz Buton")
                break

        if clicked_action:
            try:
                logging.info(f"Buton tıklandı: {clicked_label}")
                clicked_action()
                return True
            except Exception as e:
                logging.error(f"Buton eylemi hatası ('{clicked_label}'): {e}")
                return False
        else:
            return False 