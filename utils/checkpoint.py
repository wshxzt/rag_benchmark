"""
Thread-safe incremental JSON checkpoint.

Persists results key-by-key as they arrive so a crashed run can be resumed.
"""
import json
import os
import threading


class Checkpoint:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        if os.path.exists(path):
            with open(path) as f:
                self._data: dict = json.load(f)
        else:
            self._data = {}

    def done(self, key: str) -> bool:
        return key in self._data

    def record(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = value
            with open(self.path, "w") as f:
                json.dump(self._data, f)

    def data(self) -> dict:
        return dict(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return True
