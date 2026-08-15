"""Manage the bgutil PO-token server.

YouTube's SABR/GVS PO-token rollout requires a proof-of-origin token for 1080p+
downloads. The bgutil server (Brainicism/bgutil-ytdlp-pot-provider) generates
those tokens. This module starts it on demand so `fast-clip serve` brings up the
full-HD download path automatically.
"""

from __future__ import annotations

import subprocess
import time
import urllib.request
from pathlib import Path

BGUTIL_SERVER_DIR = Path.home() / "bgutil-ytdlp-pot-provider" / "server"
BGUTIL_PORT = 4416
BGUTIL_PING = f"http://127.0.0.1:{BGUTIL_PORT}/ping"


def is_running() -> bool:
    """True if the bgutil server responds on its health endpoint."""
    try:
        with urllib.request.urlopen(BGUTIL_PING, timeout=1.0) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 — any failure means not running
        return False


def ensure_bgutil_server() -> bool:
    """Start the bgutil server if it isn't up. Returns True once it's serving.

    No-op if already running. Returns False if the server isn't installed or
    fails to start — in which case downloads gracefully fall back to 360p.
    """
    if is_running():
        return True

    main_js = BGUTIL_SERVER_DIR / "build" / "main.js"
    if not main_js.exists():
        return False

    try:
        # start_new_session detaches it so it survives this process exiting.
        subprocess.Popen(
            ["node", "build/main.js"],
            cwd=str(BGUTIL_SERVER_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError):
        return False

    # Wait up to ~10s for it to come up.
    for _ in range(50):
        if is_running():
            return True
        time.sleep(0.2)
    return False
