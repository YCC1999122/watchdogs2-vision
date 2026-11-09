"""Desktop wrapper: a windowed app (not a background service).

Opens a local HTTP server on 127.0.0.1 (random port) inside this process
and shows it in a native WebView2 window. Closing the window stops the
server and exits cleanly.

Run (dev):
  runtime\\python\\python.exe desktop.py
"""
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from http.server import ThreadingHTTPServer
from server import Handler


class Api:
    """Exposed to the webview as window.pywebview.api ."""

    def open_folder(self, path):
        folder = os.path.dirname(path) if path and os.path.isfile(path) else path
        if folder and os.path.isdir(folder):
            os.startfile(folder)


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    import webview
    webview.create_window(
        "NetHack Layer View 转换",
        f"http://127.0.0.1:{port}/",
        width=760, height=660,
        js_api=Api(),
    )
    webview.start()          # blocks until window closed
    srv.shutdown()
    srv.server_close()


if __name__ == "__main__":
    main()
