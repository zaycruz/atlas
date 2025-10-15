"""Background watchers for OS events (files, clipboard)."""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Optional

from .telemetry import Telemetry  # noqa: F401 - imported for future use

LOGGER = logging.getLogger(__name__)

try:
    import pyperclip  # type: ignore
except Exception:
    pyperclip = None  # type: ignore


class FileWatcher(threading.Thread):
    """Poll directories for file changes and log episodic events."""

    def __init__(
        self,
        agent,
        directories: Iterable[Path],
        *,
        interval: float = 5.0,
        extensions: Optional[Iterable[str]] = None,
        max_size: int = 512_000,
    ) -> None:
        super().__init__(daemon=True)
        self.agent = agent
        self.directories = [Path(d).expanduser() for d in directories]
        self.interval = max(1.0, float(interval))
        self.extensions = {e.lower().lstrip(".") for e in extensions} if extensions else set()
        self.max_size = max_size
        self._stop_event = threading.Event()
        self._seen: dict[str, float] = {}

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        if not getattr(self.agent, "layered_memory", None):
            return
        tracked_dirs = [d for d in self.directories if d.exists() and d.is_dir()]
        if not tracked_dirs:
            return
        while not self._stop_event.is_set():
            for directory in tracked_dirs:
                self._scan_directory(directory)
            self._stop_event.wait(self.interval)

    def _scan_directory(self, directory: Path) -> None:
        try:
            for entry in directory.rglob("*"):
                if not entry.is_file():
                    continue
                if self.extensions and entry.suffix.lstrip(".").lower() not in self.extensions:
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                if stat.st_size > self.max_size:
                    continue
                key = str(entry)
                mtime = stat.st_mtime
                if key in self._seen and mtime <= self._seen[key]:
                    continue
                self._seen[key] = mtime
                self._record_event(entry, mtime, stat.st_size)
        except Exception:
            LOGGER.debug("File watcher scan failed", exc_info=True)

    def _record_event(self, path: Path, mtime: float, size: int) -> None:
        memory = getattr(self.agent, "layered_memory", None)
        if memory is None:
            return
        timestamp = datetime.fromtimestamp(mtime).isoformat()
        summary = f"File updated: {path.name} ({size} bytes) at {timestamp}"
        metadata = {
            "source": "watcher:file",
            "path": str(path),
            "size": size,
            "timestamp": timestamp,
        }
        try:
            memory.log_interaction("", summary, metadata=metadata)
        except Exception:
            LOGGER.debug("Failed to log file watcher event", exc_info=True)


class ClipboardWatcher(threading.Thread):
    """Poll the system clipboard for new text snippets."""

    def __init__(self, agent, *, interval: float = 4.0, minimum_length: int = 64) -> None:
        super().__init__(daemon=True)
        self.agent = agent
        self.interval = max(1.0, float(interval))
        self.minimum_length = max(8, int(minimum_length))
        self.available = pyperclip is not None
        self._stop_event = threading.Event()
        self._last_hash: Optional[str] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        if not self.available or not getattr(self.agent, "layered_memory", None):
            return
        while not self._stop_event.is_set():
            try:
                content = pyperclip.paste()
            except Exception:
                LOGGER.debug("Clipboard watcher unable to access clipboard", exc_info=True)
                break
            if isinstance(content, str):
                stripped = content.strip()
                if len(stripped) >= self.minimum_length:
                    digest = sha256(stripped.encode("utf-8")).hexdigest()
                    if digest != self._last_hash:
                        self._record_event(stripped)
                        self._last_hash = digest
            self._stop_event.wait(self.interval)

    def _record_event(self, text: str) -> None:
        memory = getattr(self.agent, "layered_memory", None)
        if memory is None:
            return
        snippet = "\n".join(text.splitlines()[:6])
        metadata = {
            "source": "watcher:clipboard",
            "length": len(text),
        }
        try:
            memory.log_interaction("", f"Clipboard captured snippet:\n{snippet}", metadata=metadata)
        except Exception:
            LOGGER.debug("Failed to log clipboard event", exc_info=True)
