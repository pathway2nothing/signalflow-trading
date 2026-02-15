"""Local development server for pipeline visualization."""

from __future__ import annotations

import http.server
import socketserver
import threading
import webbrowser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from signalflow.viz.graph import PipelineGraph


class VizServer:
    """Simple HTTP server for serving pipeline visualization."""

    def __init__(self, graph: "PipelineGraph", port: int = 4141):
        self.graph = graph
        self.port = port
        self._server: socketserver.TCPServer | None = None
        self._thread: threading.Thread | None = None

    def _create_handler(self):
        """Create request handler with graph data."""
        from signalflow.viz.renderers.html import HtmlRenderer

        html_content = HtmlRenderer(self.graph).render()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(html_content.encode("utf-8"))
                elif self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"status": "ok"}')
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Suppress default logging
                pass

        return Handler

    def _find_available_port(self) -> int:
        """Find an available port starting from self.port."""
        import socket

        for port in range(self.port, self.port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("localhost", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"No available ports in range {self.port}-{self.port + 100}")

    def start(self, open_browser: bool = True) -> str:
        """
        Start the visualization server.

        Args:
            open_browser: Whether to open browser automatically

        Returns:
            URL where visualization is served
        """
        self.port = self._find_available_port()
        handler = self._create_handler()

        self._server = socketserver.TCPServer(("localhost", self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://localhost:{self.port}"

        if open_browser:
            webbrowser.open(url)

        return url

    def stop(self):
        """Stop the server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None

    def serve_forever(self, open_browser: bool = True):
        """Start server and block until interrupted."""
        url = self.start(open_browser=open_browser)
        print(f"\n  SignalFlow Viz running at: \033[1;36m{url}\033[0m")
        print("  Press Ctrl+C to stop\n")

        try:
            while True:
                self._thread.join(timeout=1)
        except KeyboardInterrupt:
            print("\n  Shutting down...")
            self.stop()


def serve(
    graph: "PipelineGraph",
    port: int = 4141,
    open_browser: bool = True,
    block: bool = True,
) -> VizServer:
    """
    Serve pipeline visualization on localhost.

    Args:
        graph: PipelineGraph to visualize
        port: Port number (default: 4141, same as Kedro)
        open_browser: Open browser automatically
        block: Block until Ctrl+C

    Returns:
        VizServer instance

    Example:
        >>> from signalflow.viz import serve
        >>> serve(graph)  # Opens http://localhost:4141
    """
    server = VizServer(graph, port=port)

    if block:
        server.serve_forever(open_browser=open_browser)
    else:
        url = server.start(open_browser=open_browser)
        print(f"  SignalFlow Viz running at: {url}")

    return server
