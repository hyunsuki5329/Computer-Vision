import cv2 as cv
import sys
import numpy as np

# 1. 이미지 불러오기
img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

# 2. 이미지 크기 줄이기 (0.5배 축소)
# fx, fy 인자를 사용하여 가로세로 비율을 조절합니다.
img_small = cv.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)

# 3. 그레이스케일 변환 (축소된 이미지 기준)
gray = cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)

# 4. hstack을 위해 그레이스케일 이미지를 3채널로 변환
gray_3channel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 5. 원본(축소본)과 그레이스케일을 가로로 연결
combined = np.hstack((img_small, gray_3channel))

# 6. 결과 화면 출력
cv.imshow('Reduced Size - Color & Gray', combined)

# 속성 출력으로 크기 확인
print(f"Original Size: {img.shape}")
print(f"Reduced Size: {img_small.shape}")

cv.waitKey()
cv.destroyAllWindows()