import requests
import logging
from typing import Optional

class TelegramBot:
    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self._test_connection()

    def _test_connection(self):
        """Test the connection to Telegram API."""
        try:
            response = requests.get(f"{self.base_url}/getMe")
            if response.status_code == 200:
                logging.info("Telegram bot connection successful")
            else:
                logging.error(f"Telegram bot connection failed: {response.text}")
        except Exception as e:
            logging.error(f"Error testing Telegram bot connection: {e}")

    def send_message(self, chat_id: str, text: str) -> bool:
        """
        Send a message to a specific chat ID.
        
        Args:
            chat_id: The Telegram chat ID to send the message to
            text: The message text to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                logging.info(f"Telegram message sent successfully to {chat_id}")
                return True
            else:
                logging.error(f"Failed to send Telegram message: {response.text}")
                return False
        except Exception as e:
            logging.error(f"Error sending Telegram message: {e}")
            return False 