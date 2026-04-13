import json
import os
import socket


def load_xr_frames(path):
    if not path:
        raise ValueError("XR replay mode requires --xr_input.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"XR input file does not exist: {path}")

    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext == ".jsonl":
            frames = [json.loads(line) for line in f if line.strip()]
        else:
            payload = json.load(f)
            if isinstance(payload, dict):
                frames = payload.get("frames", [])
            elif isinstance(payload, list):
                frames = payload
            else:
                raise ValueError(f"Unsupported XR input payload type: {type(payload).__name__}")
    if not isinstance(frames, list):
        raise ValueError("XR input must resolve to a list of frames.")
    return frames


class SocketFrameSource:
    def __init__(self, host, port):
        self.host = host
        self.port = int(port)
        self.listener = None
        self.conn = None
        self.reader = None

    def __enter__(self):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((self.host, self.port))
        self.listener.listen(1)
        print(f"[openxr-socket] waiting for client on {self.host}:{self.port}")
        self.conn, addr = self.listener.accept()
        print(f"[openxr-socket] connected by {addr}")
        self.reader = self.conn.makefile("r", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __iter__(self):
        if self.reader is None:
            raise RuntimeError("SocketFrameSource must be used as a context manager.")
        for line in self.reader:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and payload.get("type") == "eos":
                break
            yield payload

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.listener is not None:
            self.listener.close()
            self.listener = None
