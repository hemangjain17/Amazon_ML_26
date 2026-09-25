import os
import sys
import time
import json
import logging
import traceback
import threading
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any

logger = logging.getLogger("TelegramNotifier")


class TelegramNotifier:
    """Sends status updates, heartbeats, logs, error tracebacks, and submission files via Telegram Bot API."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None, enabled: bool = True):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = enabled and bool(self.bot_token and self.chat_id)

        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        self._last_status = "Initialized"

        if self.enabled:
            print(f"📱 Telegram Notifier enabled for Chat ID: {self.chat_id}")
        else:
            print("ℹ️ Telegram Notifier disabled (Token or Chat ID missing). Logging locally only.")

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Sends a text message to the Telegram chat."""
        if not self.enabled:
            logger.info(f"[Telegram Mock] {text}")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res.get("ok", False)
        except Exception as e:
            logger.warning(f"Failed to send Telegram message: {e}")
            return False

    def send_document(self, file_path: str, caption: Optional[str] = None) -> bool:
        """Sends a document/file attachment to Telegram (e.g. TSV submission files)."""
        if not self.enabled or not os.path.exists(file_path):
            if not os.path.exists(file_path):
                logger.warning(f"File not found for Telegram send: {file_path}")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"

        boundary = f"----WebKitFormBoundary{int(time.time()*1000)}"
        body = []

        # chat_id field
        body.append(f"--{boundary}".encode("utf-8"))
        body.append(f'Content-Disposition: form-data; name="chat_id"'.encode("utf-8"))
        body.append(b"")
        body.append(str(self.chat_id).encode("utf-8"))

        # caption field
        if caption:
            body.append(f"--{boundary}".encode("utf-8"))
            body.append(f'Content-Disposition: form-data; name="caption"'.encode("utf-8"))
            body.append(b"")
            body.append(str(caption).encode("utf-8"))

        # file field
        filename = os.path.basename(file_path)
        body.append(f"--{boundary}".encode("utf-8"))
        body.append(f'Content-Disposition: form-data; name="document"; filename="{filename}"'.encode("utf-8"))
        body.append(b"Content-Type: application/octet-stream")
        body.append(b"")

        with open(file_path, "rb") as f:
            file_data = f.read()

        body_bytes = b"\r\n".join(body) + b"\r\n" + file_data + b"\r\n" + f"--{boundary}--\r\n".encode("utf-8")

        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body_bytes)),
        }

        try:
            req = urllib.request.Request(url, data=body_bytes, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res.get("ok", False)
        except Exception as e:
            logger.warning(f"Failed to send Telegram document {filename}: {e}")
            return False

    def send_start(self, pipeline_name: str, config_summary: Dict[str, Any]):
        """Sends pipeline start announcement with config details."""
        details = "\n".join([f"• <b>{k}:</b> {v}" for k, v in config_summary.items()])
        msg = (
            f"🚀 <b>[PIPELINE STARTED] {pipeline_name}</b>\n\n"
            f"<b>Configuration:</b>\n{details}\n\n"
            f"<i>Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}</i>"
        )
        self.send_message(msg)

    def send_error(self, step_name: str, exception: Exception):
        """Sends error notification with formatted stack traceback."""
        tb = traceback.format_exc()
        msg = (
            f"❌ <b>[PIPELINE ERROR] {step_name}</b>\n\n"
            f"<b>Error:</b> {str(exception)}\n\n"
            f"<b>Traceback:</b>\n<pre>{tb[-1000:]}</pre>"
        )
        self.send_message(msg)

    def send_success(self, metrics: Dict[str, Any], files: Optional[list] = None):
        """Sends success notification and uploads output files as attachments."""
        details = "\n".join([f"• <b>{k}:</b> {v}" for k, v in metrics.items()])
        msg = (
            f"🎉 <b>[PIPELINE COMPLETED SUCCESSFULLY]</b>\n\n"
            f"<b>Final Metrics:</b>\n{details}\n\n"
            f"<i>Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}</i>"
        )
        self.send_message(msg)

        if files:
            for f in files:
                if os.path.exists(f):
                    self.send_document(f, caption=f"📄 Output File: {os.path.basename(f)}")

    def start_heartbeat(self, interval_sec: int = 300, get_status_fn=None):
        """Starts a background thread to periodically send heartbeats."""
        if not self.enabled or self._heartbeat_thread is not None:
            return

        self._stop_heartbeat.clear()

        def _heartbeat_loop():
            while not self._stop_heartbeat.wait(interval_sec):
                status_text = get_status_fn() if get_status_fn else self._last_status
                msg = f"💓 <b>[HEARTBEAT] Pipeline active</b>\nStatus: {status_text}\nTime: {time.strftime('%H:%M:%S')}"
                self.send_message(msg)

        self._heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self):
        """Stops the heartbeat background thread."""
        if self._heartbeat_thread:
            self._stop_heartbeat.set()
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None

    def update_status(self, text: str):
        self._last_status = text
