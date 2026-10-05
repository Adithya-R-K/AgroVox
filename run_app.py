"""
AgroVox — Main Application Launcher
Runs the New Modern UI / UX frontend on http://localhost:3000 and opens it in your default browser.
"""
import os
import sys
import webbrowser
import http.server
import socketserver
import threading
import time

PORT = 3000
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
WEBSITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "website")

SERVE_DIR = FRONTEND_DIR if os.path.exists(FRONTEND_DIR) else WEBSITE_DIR

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    def log_message(self, format, *args):
        # Quiet standard logging to keep terminal clean
        pass

def main():
    print("=" * 70)
    print("      🌾 AgroVox — Voice-Driven Intelligence for Smarter Farming 🌾")
    print("                 [ Complete Modern UI / UX Platform ]")
    print("=" * 70)
    print(f"\n[+] Serving Modern UI from: {SERVE_DIR}")
    print(f"[+] Access URL:             http://localhost:{PORT}")
    print("\n[+] Opening browser automatically...")

    def open_browser():
        time.sleep(1.2)
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=open_browser, daemon=True).start()

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("\n" + "=" * 70)
        print(f"  🚀 Server is running! Open http://localhost:{PORT} in your browser.")
        print("  Press Ctrl+C to stop the server.")
        print("=" * 70 + "\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[!] Shutting down AgroVox server. Goodbye!")

if __name__ == "__main__":
    main()
