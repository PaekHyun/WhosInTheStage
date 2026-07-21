"""WhosInTheStage 설정 상수 모듈."""

import os

# ── 경로 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-seg.pt")

# ── YOLO ──────────────────────────────────────────────
YOLO_CONFIDENCE: float = 0.5
YOLO_CLASSES: list[int] = [0]  # COCO class 0 = person

# ── 카메라 ────────────────────────────────────────────
CAMERA_PRIORITY: list[int] = [1, 0, 2, 3, 4]

# ── 윈도우 ────────────────────────────────────────────
WINDOW_TITLE: str = "YOLO 실시간 배경 합성 매니저"
WINDOW_GEOMETRY: tuple[int, int, int, int] = (100, 100, 600, 300)
CV2_WINDOW_NAME: str = "Real-time YOLO Virtual Background"

# ── UI ────────────────────────────────────────────────
RUN_BUTTON_SIZE: tuple[int, int] = (150, 150)
RUN_BUTTON_STYLE: str = (
    "font-size: 20px; background-color: #4CAF50; color: white; font-weight: bold;"
)
