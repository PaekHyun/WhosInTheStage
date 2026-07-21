"""YOLO 세그멘테이션 기반 배경 합성 프로세서."""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional

from config import MODEL_PATH, YOLO_CONFIDENCE, YOLO_CLASSES


class BackgroundProcessor:
    """YOLOv8-seg 모델을 사용해 사람을 분리하고 배경을 합성합니다."""

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        """모델을 로드합니다.

        Args:
            model_path: YOLO 모델 가중치 파일 경로.

        Raises:
            FileNotFoundError: 모델 파일이 존재하지 않을 때.
        """
        import os

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"YOLO 모델 파일을 찾을 수 없습니다: {model_path}\n"
                f"yolov8n-seg.pt 파일을 프로젝트 루트에 배치하세요."
            )
        self.model = YOLO(model_path)

    def composite(self, frame: np.ndarray, background: np.ndarray) -> np.ndarray:
        """프레임에서 사람을 분리해 배경 위에 합성합니다.

        Args:
            frame: 웹캠 원본 프레임 (BGR).
            background: 배경 이미지 (frame과 동일 해상도, BGR).

        Returns:
            사람이 배경 위에 합성된 이미지 (BGR).
        """
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
