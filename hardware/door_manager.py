import logging
import time
from utils.helpers import handle_error
from common.constants import DOOR_OPEN_TIME
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
            "action_handler": None
        }

    def update_status(self, message, color):
        """Update the system status message and color."""
        self.system_status["status"] = message
        self.system_status["color"] = color

    def update_door_state(self, is_stable_now, liveness_passed, current_mode):
        """Update door state based on recognition and liveness status."""
        # Door opening condition
        if (current_mode == "normal" and is_stable_now and
                liveness_passed and self.controller and not self.controller.get_state()):
            logging.info("Door opening conditions met")
            if self.controller.open_door():
                self.door_opened_time = time.time()
                self.update_status("Door Opened", "green")
            else:
                self.update_status("Door Opening Error!", "red")

        # Door closing condition
        if (self.controller and self.controller.get_state() and
                self.door_opened_time and (time.time() - self.door_opened_time > DOOR_OPEN_TIME)):
            logging.info("Door open time exceeded, closing")
            if self.controller.close_door():
                self.door_opened_time = None
                self.update_status("Door Closed", "red")
                return True  # Indicate that door was closed
            else:
                self.update_status("Door Closing Error!", "red")
                logging.error("Failed to close door!")

        return False  # Door was not closed

    def open_door(self):
        """Open the door and keep it open for the specified duration."""
        if self.controller.open_door():
            logging.info("Door opened successfully")
            return True
        return False

    def close_door(self):
        """Close the door."""
        if self.controller.close_door():
            logging.info("Door closed successfully")
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