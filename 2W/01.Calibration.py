import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수 (가로 9개, 세로 6개)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표(3D) 생성: (0,0,0), (25,0,0), (50,0,0) ...
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = [] # 3D 실제 세계 좌표
imgpoints = [] # 2D 이미지 평면 좌표

# 이미지 로드
images = glob.glob("2W/calibration_images/left*.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    # 1-1. 이미지 읽기 및 그레이스케일 변환
    img = cv2.imread(fname)
    # findChessboardCorners는 연산 속도와 정확도를 위해 흑백(Grayscale) 이미지를 입력으로 받습니다.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이미지의 크기 저장 (나중에 캘리브레이션 함수에서 전체 이미지 규격을 알기 위해 사용)
    # gray.shape는 (height, width) 순서이므로 역순 [::-1]으로 취해 (width, height)로 만듭니다.
    img_size = gray.shape[::-1] 

    # 1-2. 체크보드 코너 찾기
    # CHECKERBOARD = (9, 6) : 내부 코너 점의 개수를 의미합니다.
    # ret: 성공 여부(True/False), corners: 검출된 코너들의 2D 좌표들
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너가 정상적으로 모두 발견되었다면 (9x6=54개가 다 찾아져야 ret이 True가 됨)
    if ret == True:
        # 1-3. 실제 세계 좌표 추가
        # 모든 이미지에서 체크보드는 동일한 규격이므로, 미리 만들어둔 실제 좌표(objp)를 리스트에 담습니다.
        objpoints.append(objp)
        
        # 1-4. 코너 좌표 정밀화 (Sub-pixel Accuracy)
        # findChessboardCorners가 찾은 좌표는 픽셀 단위(정수)에 가깝습니다.
        # cv2.cornerSubPix를 사용하면 수학적으로 계산하여 소수점 단위의 아주 정밀한 위치를 찾아냅니다.
        # criteria: 정밀화 계산을 언제 멈출지 결정하는 조건 (반복 횟수 또는 목표 오차)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 정밀화된 2D 이미지 좌표를 리스트에 담습니다.
        imgpoints.append(corners2)

        # 1-5. 검출 결과 시각화
        # 찾은 코너들을 이미지 위에 선과 점으로 그려줍니다.
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corner Detection', img)
        # 검출되는 과정을 확인하기 위해 잠시 대기 (0.1초)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# cv2.calibrateCamera()를 통해 K(내부 행렬), dist(왜곡 계수) 산출
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
# 테스트할 첫 번째 이미지 로드
test_img = cv2.imread(images[0])
h, w = test_img.shape[:2]

# 1. 새로운 카메라 매트릭스 계산 (검은 부분을 포함할지 결정)
# alpha=1: 모든 픽셀을 유지 (보정 후 휘어진 경계 때문에 검은 부분이 나타남)
# alpha=0: 검은 부분을 제거하고 유효한 픽셀만 남도록 잘라냄
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

# 2. 왜곡 보정 수행 (수정된 new_camera_mtx 사용)
dst = cv2.undistort(test_img, K, dist, None, new_camera_mtx)

# 3. 결과 시각화
# 원본과 왜곡 보정본을 가로로 붙여서 비교
result = np.hstack((test_img, dst))
cv2.imshow('Original vs Undistorted', result)

# 결과 저장
cv2.imwrite('2W/calibration_result.jpg', dst)
print("\n왜곡 보정 완료! 'calibration_result.jpg'로 저장되었습니다.")

cv2.waitKey(0)
cv2.destroyAllWindows()