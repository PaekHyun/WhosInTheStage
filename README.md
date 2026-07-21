# 🎭 WhosInTheStage

**YOLOv8 기반 실시간 가상 배경 합성 시스템**

웹캠 영상에서 사람을 실시간으로 분리하고, 선택한 배경 이미지와 음악을 합성하여 전체화면으로 출력하는 애플리케이션입니다. 무대·프레젠테이션·이벤트 등에서 손쉽게 가상 배경을 구성할 수 있습니다.

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 🧍 실시간 사람 분리 | YOLOv8-seg 모델로 프레임 단위 사람 세그멘테이션 |
| 🖼️ 배경 합성 | 선택한 JPG 이미지를 배경으로 실시간 합성 |
| 🎵 오디오 재생 | MP3 파일 재생과 동기화 — 음악이 끝나면 자동 종료 |
| 📺 전체화면 출력 | OpenCV 전체화면 모드로 몰입감 있는 시청 경험 |
| 🎛️ GUI 매니저 | PyQt5 기반 파일 선택 UI — 클릭만으로 설정 |

---

## 🛠️ 기술 스택

- **Python 3.10+**
- **YOLOv8-seg** (Ultralytics) — 사람 인스턴스 세그멘테이션
- **OpenCV** — 웹캠 캡처 및 이미지 합성
- **PyQt5** — 파일 선택 GUI
- **Pygame** — MP3 오디오 재생

---

## 🚀 설치 및 실행

### 1. 저장소 클론

```bash
git clone https://github.com/PaekHyun/WhosInTheStage.git
cd WhosInTheStage
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. YOLO 모델 다운로드

```bash
# Ultralytics에서 자동 다운로드되거나, 수동으로 프로젝트 폴더에 배치
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt
```

### 4. 실행

```bash
python WhosInTheStage.py
```

---

## 📁 프로젝트 구조

```
WhosInTheStage/
├── WhosInTheStage.py   # 단일 파일 (전체 로직 포함)
├── yolov8n-seg.pt      # YOLO 모델 가중치 (별도 다운로드)
├── requirements.txt    # Python 의존성
├── .gitignore          # Git 무시 규칙
└── README.md           # 프로젝트 문서
```

---

## 🎮 사용법

1. **앱 실행** → `python WhosInTheStage.py`
2. **🎵 음악 선택** → MP3 파일 선택
3. **🖼️ 배경 선택** → JPG/JPEG 배경 이미지 선택
4. **🚀 프로그램 실행** → 전체화면 실시간 합성 시작
5. **ESC 키** → 종료 (또는 음악이 끝나면 자동 종료)

---

## 📦 실행 파일 빌드 (PyInstaller)

> ⚠️ PyTorch 최신 버전은 PyInstaller와 호환성 문제가 있을 수 있습니다.
> 안정적인 빌드를 위해 **PyTorch 2.8.0** 사용을 권장합니다.

```bash
pip install torch==2.8.0
pyinstaller WhosInTheStage.py ^
  --onefile ^
  --noconsole ^
  --collect-all torch ^
  --collect-all ultralytics ^
  --add-data "yolov8n-seg.pt;."
```

---

## ⚙️ 설정 변경

`WhosInTheStage.py` 상단의 상수를 수정하여 설정을 변경할 수 있습니다:

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `YOLO_CONFIDENCE` | `0.5` | 사람 감지 신뢰도 임계값 |
| `YOLO_CLASSES` | `[0]` | 감지할 COCO 클래스 (0=person) |
| `CAMERA_PRIORITY` | `[1, 0, 2, 3, 4]` | 카메라 탐색 우선순위 |
| `MODEL_PATH` | `yolov8n-seg.pt` | YOLO 모델 파일 경로 |

---

## 📝 라이선스

이 프로젝트는 학습 및 연구 목적으로 제작되었습니다.
