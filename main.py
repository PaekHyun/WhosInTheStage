"""WhosInTheStage — YOLO 실시간 가상 배경 합성 시스템.

Usage:
    python main.py
"""

import sys

from PyQt5.QtWidgets import QApplication

from gui import VirtualBackgroundApp


def main() -> int:
    """애플리케이션을 시작합니다."""
    app = QApplication(sys.argv)
    window = VirtualBackgroundApp()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
