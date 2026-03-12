import cv2
import numpy as np


# 1. 이미지 로드
img = cv2.imread('2W/rose.png')

# 이미지의 세로(h), 가로(w) 크기를 가져옵니다. 
# 변환 시 기준점이 될 이미지의 정중앙(center) 좌표를 계산합니다.
h, w = img.shape[:2]
center = (w // 2, h // 2) 

# 2. 회전 및 크기 조절 행렬 생성
# cv2.getRotationMatrix2D 함수는 (중심점, 회전각도, 배율)을 입력받아 2x3 행렬 M을 만듭니다.
angle = 30  # 반시계 방향으로 30도 회전
scale = 0.8  # 원본 이미지의 80% 크기로 축소
M = cv2.getRotationMatrix2D(center, angle, scale)


# 3. 평행이동 반영
# 변환 행렬 M의 구조는 다음과 같습니다:
# 여기서 M[0, 2]는 x축 평행이동(tx), M[1, 2]는 y축 평행이동(ty)을 담당

# 요구사항: x축 방향으로 +80px 이동 (오른쪽으로 이동)
M[0, 2] += 80
# 요구사항: y축 방향으로 -40px 이동 (위쪽으로 이동)
M[1, 2] += -40

# 4. Affine 변환 적용
# cv2.warpAffine은 계산된 행렬 M을 이미지의 모든 픽셀에 적용하여 실제 변환을 수행합니다.
# 마지막 인자 (w, h)는 결과 이미지의 출력 크기를 결정합니다.
dst = cv2.warpAffine(img, M, (w, h))

# ---------------------------------------------------------
# 5. 시각화
# ---------------------------------------------------------

# 원본 이미지를 보여주는 창
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', img)

# 변환(회전+축소+이동)이 완료된 이미지를 보여주는 창
cv2.namedWindow('Transformed Image', cv2.WINDOW_NORMAL)
cv2.imshow('Transformed Image', dst)

# 터미널 창에 현재 적용된 수치를 출력하여 확인합니다.
print(f"적용 내용: 회전({angle}도), 크기({scale}배), 이동(x:+80, y:-40)")

# 변환된 이미지를 파일로 저장합니다.
cv2.imwrite('2W/transformed_rose.png', dst)

# 아무 키나 누를 때까지 창을 유지하다가, 키 입력이 있으면 모든 창을 닫고 프로그램을 종료합니다.
cv2.waitKey(0)
cv2.destroyAllWindows()