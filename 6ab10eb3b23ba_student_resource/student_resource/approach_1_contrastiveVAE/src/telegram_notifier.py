import json
import os
import time
import threading
import traceback
import urllib.request
from typing import Callable, Optional


class TelegramNotifier:
    """Small Telegram Bot API client for Kaggle progress and artifact notifications."""

    MAX_DOCUMENT_BYTES = 50 * 1024 * 1024

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self._stop = threading.Event()
        self._thread = None
        self._status = "initialized"

    def send_message(self, text: str) -> bool:
        if not self.enabled:
            print(f"[Telegram disabled] {text}")
            return False
        payload = json.dumps({"chat_id": self.chat_id, "text": text}).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8")).get("ok", False)
        except Exception as exc:
            print(f"Telegram message failed: {exc}")
            return False

    def send_document(self, path: str, caption: str = "") -> bool:
        if not os.path.exists(path):
            self.send_message(f"Artifact missing: {path}")
            return False
        size = os.path.getsize(path)
        if size > self.MAX_DOCUMENT_BYTES:
            self.send_message(
                f"Artifact is {size / 1024 / 1024:.1f} MB, above Telegram's 50 MB limit; "
                f"saved locally at: {path}"
            )
            return False
        if not self.enabled:
            print(f"[Telegram disabled] artifact ready: {path}")
            return False

        boundary = f"----KaggleTelegram{int(time.time() * 1000)}"
        with open(path, "rb") as handle:
            file_bytes = handle.read()
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{self.chat_id}\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"document\"; "
                f"filename=\"{os.path.basename(path)}\"\r\n"
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode() + file_bytes + b"\r\n",
        ]
        if caption:
            parts.insert(1, f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode())
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/sendDocument",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8")).get("ok", False)
        except Exception as exc:
            self.send_message(f"Artifact upload failed for {os.path.basename(path)}: {exc}")
            return False

    def start_heartbeat(self, interval_seconds: int = 300, status: Optional[Callable[[], str]] = None):
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()

        def loop():
            while not self._stop.wait(interval_seconds):
                current = status() if status else self._status
                self.send_message(f"HEARTBEAT: Approach 1 still running. Status: {current}")

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_heartbeat(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def update(self, status: str):
        self._status = status

    def error(self, stage: str, exc: Exception):
        self.send_message(f"ERROR in {stage}: {exc}\n\n{traceback.format_exc()[-2500:]}")
