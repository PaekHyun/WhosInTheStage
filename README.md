# WhosInTheStage
Real-time system that composites a live webcam person into a picture and plays audio file the merged result instantly.

Using a YOLO model, it separates a person from a live webcam feed and composites the result with pre-saved JPEG and MP3 files on the PC, then outputs the combined result in real time.

Since PyInstaller does not work properly with the latest versions of PyTorch, it is recommended to install version 2.8.0 when creating an executable (.exe) file.

yolo 모델을 이용하여 실시간으로 웹캠 영상에서 사람을 분리하고, 
저장해둔 PC의 jpeg와 mp3파일을 합쳐서 실시간으로 출력해줍니다.
pytorch 최신 버전의 경우, pyinstaller가 제대로 동작하지 않기 때문에 
exe파일로 만들 경우에는 2.8.0으로 설치하길 추천드립니다.