import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def main():
    # 1) 입력 이미지 준비
    # 스크립트 기준으로 입력 이미지 경로 구성
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "dabo.jpg")

    # 과제 요구사항: cv.imread()로 이미지 읽기
    original_bgr = cv.imread(image_path)

    # 한글 경로 등으로 imread가 실패할 때를 대비한 보조 로딩
    if original_bgr is None:
        raw = np.fromfile(image_path, dtype=np.uint8)
        if raw.size > 0:
            original_bgr = cv.imdecode(raw, cv.IMREAD_COLOR)

    # 이미지가 끝까지 로드되지 않으면 즉시 예외 발생
    if original_bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    # 2) 에지와 직선 검출
    # 캐니 에지 검출을 위해 그레이스케일로 변환
    gray = cv.cvtColor(original_bgr, cv.COLOR_BGR2GRAY)

    # 과제 힌트 기준 threshold1=100, threshold2=200
    edges = cv.Canny(gray, 100, 200)

    # 허프 확률 직선 변환으로 선분 검출
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10,
    )

    # 원본 복사본에 검출된 직선을 빨간색(BGR: 0, 0, 255), 두께 2로 표시
    line_image = original_bgr.copy() # 검출된 선분을 그릴 이미지 복사본
    if lines is not None:
        # HoughLinesP 결과는 [ [x1,y1,x2,y2] ] 형태이므로 순회하며 선분 그리기
        for line in lines:
            x1, y1, x2, y2 = line[0] # line[0]에서 좌표 추출
            cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2) # 빨간색 선 그리기

    # 3) 시각화 준비
    # matplotlib 표시를 위해 BGR -> RGB 변환
    original_rgb = cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB)
    line_rgb = cv.cvtColor(line_image, cv.COLOR_BGR2RGB)

    # 원본 이미지와 직선 검출 결과를 나란히 시각화
    plt.figure(figsize=(12, 5))

    # 왼쪽: 원본
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # 오른쪽: 직선 검출 결과
    plt.subplot(1, 2, 2)
    plt.imshow(line_rgb)
    plt.title("Detected Lines (Canny + HoughLinesP)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
