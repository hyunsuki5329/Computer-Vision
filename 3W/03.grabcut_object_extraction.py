import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def main():
    # 1) 입력 이미지 준비
    # 스크립트 기준으로 입력 이미지 경로 구성
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "coffee cup.jpg")

    # 원본 이미지 읽기
    original_bgr = cv.imread(image_path)

    # 한글 경로 등으로 imread가 실패할 때를 대비한 보조 로딩
    if original_bgr is None:
        raw = np.fromfile(image_path, dtype=np.uint8)
        if raw.size > 0:
            original_bgr = cv.imdecode(raw, cv.IMREAD_COLOR)

    # 이미지가 끝까지 로드되지 않으면 즉시 예외 발생
    if original_bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    # 2) GrabCut 분할 준비
    # GrabCut용 마스크와 모델 초기화
    mask = np.zeros(original_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 이미지 가장자리 배경이 포함되도록 내부 영역을 초기 전경 후보로 지정
    # 초기 사각형 영역 설정: (x, y, width, height)
    h, w = original_bgr.shape[:2]
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))

    # 3) GrabCut 실행
    # 사각형 기반 GrabCut 수행
    cv.grabCut(
        original_bgr,
        mask,
        rect,
        bgd_model,
        fgd_model,
        iterCount=5,
        mode=cv.GC_INIT_WITH_RECT,
    )

    # 4) 이진 마스크 생성
    # 배경/전경 라벨을 이진 마스크(0, 1)로 변환
    mask_binary = np.where(
        (mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),
        0,
        1,
    ).astype("uint8")

    # 5) 배경 제거(전경만 남기기)
    # 원본 이미지에 마스크를 곱해 배경 제거
    foreground_bgr = original_bgr * mask_binary[:, :, np.newaxis]

    # 6) 시각화 준비
    # matplotlib 시각화를 위해 BGR을 RGB로 변환
    original_rgb = cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB)
    foreground_rgb = cv.cvtColor(foreground_bgr, cv.COLOR_BGR2RGB)

    # 원본, 마스크, 배경 제거 결과를 나란히 출력
    plt.figure(figsize=(15, 5))

    # 왼쪽: 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # 가운데: GrabCut 이진 마스크
    plt.subplot(1, 3, 2)
    plt.imshow(mask_binary, cmap="gray")
    plt.title("GrabCut Mask")
    plt.axis("off")

    # 오른쪽: 배경 제거 결과
    plt.subplot(1, 3, 3)
    plt.imshow(foreground_rgb)
    plt.title("Foreground Only")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
