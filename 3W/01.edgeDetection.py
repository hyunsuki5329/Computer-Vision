import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def main():
	# 1) 입력 이미지 준비
	# 실행 중인 스크립트 위치를 기준으로 이미지 경로를 구성
	script_dir = os.path.dirname(os.path.abspath(__file__))
	image_path = os.path.join(script_dir, "edgeDetectionImage.jpg")

	# cv.imread()로 이미지 읽기
	original_bgr = cv.imread(image_path)

	# 한글 경로 등으로 imread가 실패할 때를 대비한 보조 로딩
	if original_bgr is None:
		raw = np.fromfile(image_path, dtype=np.uint8)
		if raw.size > 0:
			original_bgr = cv.imdecode(raw, cv.IMREAD_COLOR)

	# 이미지가 끝까지 로드되지 않으면 즉시 예외 발생
	if original_bgr is None:
		raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

	# 2) 에지 계산
	# BGR 원본을 그레이스케일로 변환
	gray = cv.cvtColor(original_bgr, cv.COLOR_BGR2GRAY)

	# Sobel 필터로 x, y 방향의 에지 성분 계산
	sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3) # x 방향 에지
	sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3) # y 방향 에지

	# x, y 성분으로부터 에지 크기(magnitude) 계산 후 uint8 변환
	edge_magnitude = cv.magnitude(sobel_x, sobel_y) # 에지 크기 계산
	edge_uint8 = cv.convertScaleAbs(edge_magnitude) # uint8 타입으로 변환 (0-255 범위로 스케일링)

	# 3) 시각화 준비
	# matplotlib 출력용으로 BGR을 RGB로 변환
	original_rgb = cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB)

	# 원본 이미지와 에지 강도 이미지를 나란히 시각화
	plt.figure(figsize=(12, 5))

	# 왼쪽: 원본
	plt.subplot(1, 2, 1)
	plt.imshow(original_rgb)
	plt.title("Original Image")
	plt.axis("off")

	# 오른쪽: Sobel 에지 강도
	plt.subplot(1, 2, 2)
	plt.imshow(edge_uint8, cmap="gray")
	plt.title("Sobel Edge Magnitude")
	plt.axis("")

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()
