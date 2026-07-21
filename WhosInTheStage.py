"""기존 WhosInTheStage.py 호환 진입점.

이 파일은 하위 호환성을 위해 유지됩니다.
새로운 진입점은 main.py를 사용하세요.
"""

from main import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
