"""
WhosInTheStage — YOLO 실시간 가상 배경 합성 시스템
====================================================

웹캠 영상에서 사람을 실시간으로 분리하고, 선택한 배경 이미지와 음악을
합성하여 전체화면으로 출력하는 애플리케이션입니다.

PyInstaller 빌드 명령:
    pyinstaller WhosInTheStage.py ^
      --onefile ^
      --noconsole ^
      --collect-all torch ^
      --collect-all ultralytics ^
      --add-data "yolov8n-seg.pt;."
"""

import os
import sys

import cv2
import numpy as np
import pygame
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════
# 설정 상수
# ═══════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-seg.pt")

YOLO_CONFIDENCE: float = 0.5
YOLO_CLASSES: list[int] = [0]  # COCO class 0 = person

CAMERA_PRIORITY: list[int] = [1, 0, 2, 3, 4]

WINDOW_TITLE: str = "YOLO 실시간 배경 합성 매니저"
WINDOW_GEOMETRY: tuple[int, int, int, int] = (100, 100, 600, 300)
CV2_WINDOW_NAME: str = "Real-time YOLO Virtual Background"

RUN_BUTTON_SIZE: tuple[int, int] = (150, 150)
RUN_BUTTON_STYLE: str = (
    "font-size: 20px; background-color: #4CAF50; color: white; font-weight: bold;"
)


# ═══════════════════════════════════════════════════════
# 카메라 자동 탐색
# ═══════════════════════════════════════════════════════

def find_camera(priority: list[int] | None = None) -> int | None:
    """사용 가능한 첫 번째 웹캠 인덱스를 반환합니다.

    Args:
        priority: 탐색할 카메라 인덱스 우선순위 목록.
                  None이면 CAMERA_PRIORITY 사용.

    Returns:
        사용 가능한 카메라 인덱스, 없으면 None.
    """
    if priority is None:
        priority = CAMERA_PRIORITY

    for index in priority:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index
        cap.release()

    return None


# ═══════════════════════════════════════════════════════
# YOLO 세그멘테이션 배경 합성
# ═══════════════════════════════════════════════════════

class BackgroundProcessor:
    """YOLOv8-seg 모델로 사람을 분리하고 배경을 합성합니다."""

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"YOLO 모델 파일을 찾을 수 없습니다: {model_path}\n"
                f"yolov8n-seg.pt 파일을 실행 파일과 같은 폴더에 배치하세요."
            )
        self.model = YOLO(model_path)

    def composite(self, frame: np.ndarray, background: np.ndarray) -> np.ndarray:
        """프레임에서 사람을 분리해 배경 위에 합성합니다."""
        height, width = frame.shape[:2]
        results = self.model.predict(
            frame, classes=YOLO_CLASSES, conf=YOLO_CONFIDENCE, verbose=False
        )

        combined = background.copy()

        if results and results[0].masks is not None:
            full_mask = np.zeros((height, width), dtype=np.uint8)
            for mask in results[0].masks.data:
                m = mask.cpu().numpy()
                m = cv2.resize(m, (width, height))
                full_mask = cv2.bitwise_or(full_mask, (m * 255).astype(np.uint8))

            mask_bool = full_mask > 0
            combined[mask_bool] = frame[mask_bool]

        return combined


# ═══════════════════════════════════════════════════════
# PyQt5 GUI
# ═══════════════════════════════════════════════════════

class VirtualBackgroundApp(QWidget):
    """YOLO 실시간 배경 합성 매니저 메인 윈도우."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.mp3_path: str = ""
        self.jpg_path: str = ""
        self._processor: BackgroundProcessor | None = None
        self._init_ui()

    # ── UI 초기화 ──────────────────────────────────────

    def _init_ui(self) -> None:
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(*WINDOW_GEOMETRY)

        main_layout = QHBoxLayout()

        # 왼쪽: 파일 선택
        left_layout = QVBoxLayout()

        self.label_mp3 = QLabel("MP3 파일을 선택하세요")
        self.label_mp3.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        btn_mp3 = QPushButton("🎵 음악 선택 (.mp3)")
        btn_mp3.clicked.connect(self._select_mp3)

        self.label_jpg = QLabel("JPEG 파일을 선택하세요")
        self.label_jpg.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        btn_jpg = QPushButton("🖼️ 배경 선택 (.jpg)")
        btn_jpg.clicked.connect(self._select_jpg)

        left_layout.addWidget(btn_mp3)
        left_layout.addWidget(self.label_mp3)
        left_layout.addSpacing(20)
        left_layout.addWidget(btn_jpg)
        left_layout.addWidget(self.label_jpg)

        # 오른쪽: 실행 버튼
        right_layout = QVBoxLayout()
        self.btn_run = QPushButton("🚀 프로그램\n실행")
        self.btn_run.setStyleSheet(RUN_BUTTON_STYLE)
        self.btn_run.setFixedSize(*RUN_BUTTON_SIZE)
        self.btn_run.clicked.connect(self._run_process)
        right_layout.addWidget(self.btn_run, alignment=Qt.AlignmentFlag.AlignCenter)

        main_layout.addLayout(left_layout, stretch=2)
        main_layout.addLayout(right_layout, stretch=1)
        self.setLayout(main_layout)

    # ── 파일 선택 ──────────────────────────────────────

    def _select_mp3(self) -> None:
        file, _ = QFileDialog.getOpenFileName(
            self, "음악 파일 선택", os.getcwd(), "Audio Files (*.mp3)"
        )
        if file:
            self.mp3_path = file
            self.label_mp3.setText(os.path.basename(file))

    def _select_jpg(self) -> None:
        file, _ = QFileDialog.getOpenFileName(
            self, "배경 이미지 선택", os.getcwd(), "Image Files (*.jpg *.jpeg)"
        )
        if file:
            self.jpg_path = file
            self.label_jpg.setText(os.path.basename(file))

    # ── 실행 ───────────────────────────────────────────

    def _get_processor(self) -> BackgroundProcessor:
        """BackgroundProcessor 싱글톤 (지연 로딩)."""
        if self._processor is None:
            self._processor = BackgroundProcessor()
        return self._processor

    def _run_process(self) -> None:
        # 입력 검증
        if not self.mp3_path or not self.jpg_path:
            QMessageBox.warning(self, "경고", "MP3와 JPG 파일을 모두 선택하세요.")
            return

        # 카메라 탐지
        cam_index = find_camera()
        if cam_index is None:
            QMessageBox.critical(self, "오류", "카메라를 찾을 수 없습니다.")
            return

        # 프로세서 로드
        try:
            processor = self._get_processor()
        except FileNotFoundError as e:
            QMessageBox.critical(self, "오류", str(e))
            return

        self._start_loop(cam_index, processor)

    def _start_loop(self, cam_index: int, processor: BackgroundProcessor) -> None:
        """메인 루프: 카메라 읽기 → 합성 → 전체화면 출력."""
        cam = cv2.VideoCapture(cam_index)
        if not cam.isOpened():
            QMessageBox.critical(self, "오류", "카메라를 열 수 없습니다.")
            return

        try:
            width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

            background_img = cv2.imread(self.jpg_path)
            if background_img is None:
                QMessageBox.critical(self, "오류", "배경 이미지를 불러올 수 없습니다.")
                return
            background_img = cv2.resize(background_img, (width, height))

            # 오디오 재생
            pygame.mixer.init()
            pygame.mixer.music.load(self.mp3_path)
            pygame.mixer.music.play()

            # 전체화면 윈도우
            cv2.namedWindow(CV2_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                CV2_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

            # 실시간 루프
            while True:
                if not pygame.mixer.music.get_busy():
                    break

                ret, frame = cam.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
                combined = processor.composite(frame, background_img)
                cv2.imshow(CV2_WINDOW_NAME, combined)

                # ESC 키로 종료
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        finally:
            cam.release()
            cv2.destroyAllWindows()
            pygame.mixer.music.stop()


# ═══════════════════════════════════════════════════════
# 진입점
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VirtualBackgroundApp()
    window.show()
    sys.exit(app.exec())
