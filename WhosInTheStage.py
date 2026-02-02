import sys
import cv2
import numpy as np
import os
from ultralytics import YOLO
import pygame
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFileDialog, QFrame)
from PyQt5.QtCore import Qt

# pyinstaller build command :
# pyinstaller WhosInTheStage.py ^
#  --onefile ^
#  --noconsole ^
#  --collect-all torch ^
#  --collect-all ultralytics ^
#  --add-data "yolov8n-seg.pt;."



# =========================
# 웹캠 자동 탐색
# =========================
def find_camera(max_index=5):
    import cv2
    # for i in range(max_index):
    for i in [1, 0, 2, 3, 4] :
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    return None


class VirtualBackgroundApp(QWidget):
    def __init__(self):
        super().__init__()
        self.mp3_path = ""
        self.jpg_path = ""
        self.initUI()
        


    def initUI(self):
        self.setWindowTitle('YOLO 실시간 배경 합성 매니저')
        self.setGeometry(100, 100, 600, 300)

        # 메인 레이아웃 (좌우 분할)
        main_layout = QHBoxLayout()

        # --- 왼쪽 영역: 파일 탐색기 및 선택 ---
        left_layout = QVBoxLayout()
        
        self.label_mp3 = QLabel("MP3 파일을 선택하세요")
        self.label_mp3.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        btn_mp3 = QPushButton("🎵 음악 선택 (.mp3)")
        btn_mp3.clicked.connect(self.select_mp3)

        self.label_jpg = QLabel("JPEG 파일을 선택하세요")
        self.label_jpg.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        btn_jpg = QPushButton("🖼️ 배경 선택 (.jpg)")
        btn_jpg.clicked.connect(self.select_jpg)

        left_layout.addWidget(btn_mp3)
        left_layout.addWidget(self.label_mp3)
        left_layout.addSpacing(20)
        left_layout.addWidget(btn_jpg)
        left_layout.addWidget(self.label_jpg)

        # --- 오른쪽 영역: 실행 버튼 ---
        right_layout = QVBoxLayout()
        btn_run = QPushButton("🚀 프로그램\n실행")
        btn_run.setStyleSheet("font-size: 20px; background-color: #4CAF50; color: white; font-weight: bold;")
        btn_run.setFixedSize(150, 150)
        btn_run.clicked.connect(self.run_process)
        right_layout.addWidget(btn_run, alignment=Qt.AlignmentFlag.AlignCenter)

        # 레이아웃 합치기
        main_layout.addLayout(left_layout, stretch=2)
        main_layout.addLayout(right_layout, stretch=1)
        self.setLayout(main_layout)

    def select_mp3(self):
        file, _ = QFileDialog.getOpenFileName(self, "음악 파일 선택", os.getcwd(), "Audio Files (*.mp3)")
        if file:
            self.mp3_path = file
            self.label_mp3.setText(file.split('/')[-1])

    def select_jpg(self):
        file, _ = QFileDialog.getOpenFileName(self, "배경 이미지 선택", os.getcwd(), "Image Files (*.jpg *.jpeg)")
        if file:
            self.jpg_path = file
            self.label_jpg.setText(file.split('/')[-1])

    def run_process(self):
        if not self.mp3_path or not self.jpg_path:
            print("파일이 모두 선택되지 않았습니다.")
            return

        # 이전 답변의 로직 실행
        self.start_yolo_background(self.mp3_path, self.jpg_path)

    def start_yolo_background(self, mp3, jpg):
        model = YOLO('yolov8n-seg.pt')
        
        pygame.mixer.init()
        pygame.mixer.music.load(mp3)
        pygame.mixer.music.play()

        background_img = cv2.imread(jpg)
        
        cam_index = find_camera()
        if cam_index is None:
            print("❌ 카메라를 찾을 수 없음")
            return
        
        cam = cv2.VideoCapture(cam_index)

        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        background_img = cv2.resize(background_img, (width, height))

        window_name = "Real-time YOLO Virtual Background"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            if not pygame.mixer.music.get_busy():
                break

            ret, frame = cam.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)

            results = model.predict(frame, classes=0, conf=0.5, verbose=False)
            combined_img = background_img.copy()

            if len(results) > 0 and results[0].masks is not None:
                full_mask = np.zeros((height, width), dtype=np.uint8)
                for mask in results[0].masks.data:
                    m = mask.cpu().numpy()
                    m = cv2.resize(m, (width, height))
                    full_mask = cv2.bitwise_or(full_mask, (m * 255).astype(np.uint8))

                mask_bool = full_mask > 0
                combined_img[mask_bool] = frame[mask_bool]

            cv2.imshow(window_name, combined_img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cam.release()
        cv2.destroyAllWindows()
        pygame.mixer.music.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VirtualBackgroundApp()
    ex.show()
    sys.exit(app.exec())