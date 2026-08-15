"""FastClip GUI entry point."""

from __future__ import annotations

import sys


def main() -> int:
    """Launch the FastClip Qt application."""
    from PySide6.QtWidgets import QApplication

    from fast_clip.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("FastClip")
    app.setOrganizationName("FastClip")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())