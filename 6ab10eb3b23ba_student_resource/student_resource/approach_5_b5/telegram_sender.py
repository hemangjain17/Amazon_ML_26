"""Best-effort, ordered Telegram delivery for large TSV artifacts."""
from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from pathlib import Path


TELEGRAM_LIMIT = 45 * 1024 * 1024


def _send_document(token: str, chat_id: str, path: Path, caption: str) -> bool:
    boundary = "----Approach5TelegramBoundary"
    with path.open("rb") as handle:
        payload = handle.read()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n{chat_id}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"caption\"\r\n\r\n{caption}\r\n".encode(),
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"document\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        ).encode(),
        payload,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendDocument",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8")).get("ok", False)
    except Exception as exc:
        print(f"Telegram upload failed for {path.name}: {exc}")
        return False


def _chunks(path: Path, chunk_bytes: int):
    with path.open("rb") as source:
        header = source.readline()
        part = 0
        while True:
            handle = tempfile.NamedTemporaryFile(
                mode="wb", prefix=f"{path.stem}_part_", suffix=path.suffix, delete=False
            )
            temp_path = Path(handle.name)
            size = 0
            wrote_data = False
            exhausted = False
            try:
                handle.write(header)
                size += len(header)
                while size < chunk_bytes:
                    line = source.readline()
                    if not line:
                        exhausted = True
                        break
                    handle.write(line)
                    size += len(line)
                    wrote_data = True
                handle.close()
                if not wrote_data and exhausted:
                    temp_path.unlink(missing_ok=True)
                    return
                part += 1
                yield part, temp_path
            except Exception:
                handle.close()
                temp_path.unlink(missing_ok=True)
                raise
            if exhausted:
                return


def send_artifacts(matching_path: Path, candidate_path: Path, chunk_mb: int = 45) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing")
        return False

    chunk_bytes = max(1, chunk_mb) * 1024 * 1024
    all_ok = True
    # Deliberately ordered: matching results first, candidate pairs second.
    for label, artifact in (("matching results", matching_path), ("candidate pairs", candidate_path)):
        artifact = Path(artifact)
        if not artifact.is_file():
            print(f"Telegram skipped missing artifact: {artifact}")
            all_ok = False
            continue
        total = artifact.stat().st_size
        print(f"Telegram sending {label}: {artifact.name} ({total / 1024 / 1024:.1f} MB)")
        for part, temporary_path in _chunks(artifact, chunk_bytes):
            try:
                caption = f"Approach 5 {label}: {artifact.name} part {part}"
                ok = _send_document(token, chat_id, temporary_path, caption)
                all_ok = all_ok and ok
                print(f"Telegram sent {label} part {part}: {'ok' if ok else 'failed'}")
            finally:
                temporary_path.unlink(missing_ok=True)
    return all_ok
