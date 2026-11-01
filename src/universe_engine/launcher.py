"""One-click launcher: starts the observatory server on a free local port and opens the browser. Used by the packaged .exe."""
import socket
import sys
import threading
import time
import webbrowser


def free_port(preferred=8000):
    for p in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p)); return s.getsockname()[1]
            except OSError:
                continue


def main():
    import uvicorn
    from .observatory.server import create_app
    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    print(f"Universe Engine running at {url}\nClose this window to stop it.", flush=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app("runs/observatory"), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
