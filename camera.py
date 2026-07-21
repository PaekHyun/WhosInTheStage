"""카메라 탐지 및 관리 유틸리티."""

from __future__ import annotations

import cv2
from typing import Optional

from config import CAMERA_PRIORITY


def find_camera(priority: list[int] | None = None) -> Optional[int]:
    """사용 가능한 첫 번째 웹캠 인덱스를 반환합니다.

    Args:
        priority: 탐색할 카메라 인덱스 우선순위 목록.
                  None이면 기본 CAMERA_PRIORITY 사용.

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
