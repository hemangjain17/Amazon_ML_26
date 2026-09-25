import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.telegram_notifier import TelegramNotifier


class TestTelegramNotifier(unittest.TestCase):

    def test_mock_telegram_notifier(self):
        """Tests TelegramNotifier gracefully handles disabled/mock mode without raising errors."""
        notifier = TelegramNotifier(enabled=False)
        self.assertFalse(notifier.enabled)

        # Ensure calling methods in disabled mode returns False smoothly
        res_msg = notifier.send_message("Test message")
        self.assertFalse(res_msg)

        res_doc = notifier.send_document("non_existent_file.tsv")
        self.assertFalse(res_doc)

        # Ensure start and error notifications don't crash
        notifier.send_start("Test Pipeline", {"param1": "val1"})
        notifier.send_error("Test Step", ValueError("Test error message"))
        notifier.send_success({"metric1": "1.0"})


if __name__ == "__main__":
    unittest.main()
