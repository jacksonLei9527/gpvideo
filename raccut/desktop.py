"""Desktop entry: run the FastAPI backend in-process and wrap the UI in a
native window via pywebview. Double-click = desktop app.

Usage:  .venv/bin/python desktop.py
"""
import socket
import threading
import time

import uvicorn
import webview

from app.main import app


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    port = free_port()
    server = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port,
                                   log_level="warning"),
        daemon=True)
    server.start()

    # wait until the server accepts connections
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            break
        except OSError:
            time.sleep(0.1)

    webview.create_window("raccut — 賽車集錦剪輯",
                          f"http://127.0.0.1:{port}",
                          width=980, height=860, min_size=(760, 600))
    webview.start()  # blocks until window closes; daemon thread dies with it


if __name__ == "__main__":
    main()
