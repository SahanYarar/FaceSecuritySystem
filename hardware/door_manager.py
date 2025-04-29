import logging
import time
from utils.helpers import handle_error
from utils.telegram_bot import TelegramBot
from common.constants import DOOR_OPEN_TIME, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from .door_controller import DoorController

class DoorManager:
    def __init__(self, door_controller=None):
        self.controller = door_controller if door_controller else DoorController()
        self.is_simulated = not self.controller.gpio_available
        if self.is_simulated:
            logging.warning("Running in simulation mode - door control will be simulated")
        self.door_opened_time = None
        self.system_status = {
            "status": "Starting...",
            "color": None,
            "liveness": "Waiting",
            "liveness_color": None,
            "action_handler": None,
            "door_remaining_time": None  # New field for remaining time
        }
        self.telegram_bot = TelegramBot(TELEGRAM_BOT_TOKEN)

    def _update_remaining_time(self):
        """Update the remaining time until door closes."""
        if self.door_opened_time and self.controller.get_state():
            elapsed = time.time() - self.door_opened_time
            remaining = max(0, DOOR_OPEN_TIME - elapsed)
            self.system_status["door_remaining_time"] = remaining
        else:
            self.system_status["door_remaining_time"] = None

    def update_status(self, message, color):
        """Update the system status message and color."""
        self.system_status["status"] = message
        self.system_status["color"] = color
        self._update_remaining_time()

    def update_door_state(self, is_stable_now, liveness_passed, current_mode, person_name=None):
        """Update door state based on recognition and liveness status."""
        # Door opening condition
        if (current_mode == "normal" and is_stable_now and
                liveness_passed and self.controller and not self.controller.get_state()):
            logging.info("Door opening conditions met")
            if self.controller.open_door():
                self.door_opened_time = time.time()
                self.update_status("Door Opened", "green")
                # Send Telegram notification with person's name
                message = f"<b>Door Opened</b>\nThe security door has been opened."
                if person_name:
                    message = f"<b>Door Opened</b>\n{person_name} has opened the security door."
                self.telegram_bot.send_message(
                    TELEGRAM_CHAT_ID,
                    message
                )
            else:
                self.update_status("Door Opening Error!", "red")

        # Door closing condition
        if (self.controller and self.controller.get_state() and
                self.door_opened_time and (time.time() - self.door_opened_time > DOOR_OPEN_TIME)):
            logging.info("Door open time exceeded, closing")
            if self.controller.close_door():
                self.door_opened_time = None
                self.update_status("Door Closed", "red")
                # Send Telegram notification with person's name
                message = f"<b>Door Closed</b>\nThe security door has been closed."
                if person_name:
                    message = f"<b>Door Closed</b>\nThe security door has been closed after {person_name}'s entry."
                self.telegram_bot.send_message(
                    TELEGRAM_CHAT_ID,
                    message
                )
                return True  # Indicate that door was closed
            else:
                self.update_status("Door Closing Error!", "red")
                logging.error("Failed to close door!")

        self._update_remaining_time()
        return False  # Door was not closed

    def open_door(self, person_name=None):
        """Open the door and keep it open for the specified duration."""
        if self.controller.open_door():
            logging.info("Door opened successfully")
            self.door_opened_time = time.time()
            self._update_remaining_time()
            # Send Telegram notification with person's name
            message = f"<b>Door Opened</b>\nThe security door has been opened."
            if person_name:
                message = f"<b>Door Opened</b>\n{person_name} has opened the security door."
            self.telegram_bot.send_message(
                TELEGRAM_CHAT_ID,
                message
            )
            return True
        return False

    def close_door(self, person_name=None):
        """Close the door."""
        if self.controller.close_door():
            logging.info("Door closed successfully")
            self.door_opened_time = None
            self._update_remaining_time()
            # Send Telegram notification with person's name
            message = f"<b>Door Closed</b>\nThe security door has been closed."
            if person_name:
                message = f"<b>Door Closed</b>\nThe security door has been closed after {person_name}'s entry."
            self.telegram_bot.send_message(
                TELEGRAM_CHAT_ID,
                message
            )
            return True
        return False

    def get_state(self):
        """Get current door state."""
        return self.controller.get_state()

    def cleanup(self):
        """Clean up resources."""
        if self.controller:
            self.controller.cleanup()
            self.controller = None 