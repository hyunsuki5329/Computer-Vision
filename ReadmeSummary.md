## 1) 이미지 로드 후 화면 표시 크기로 축소

큰 원본 이미지를 그대로 띄우면 창이 너무 커질 수 있어 먼저 축소합니다.
이 단계에서 이후 처리 대상의 해상도가 결정됩니다.

```python
img = cv.imread('soccer.jpg')  # 원본 이미지 로드
img_small = cv.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)  # 가로/세로 0.5배 축소
```

## 2) 그레이스케일 변환 후 hstack 가능한 형태로 정리

그레이스케일은 1채널이므로 컬러 영상(3채널)과 바로 이어붙일 수 없습니다.
따라서 COLOR_GRAY2BGR로 3채널로 맞춘 뒤 np.hstack을 수행합니다.

```python
gray = cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)  # 흑백 변환
gray_3channel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  # 1채널 -> 3채널 변환
combined = np.hstack((img_small, gray_3channel))  # 좌우 비교용 결합
```

## 3) 결합 영상 출력 및 크기 정보 확인

결과 영상을 띄우고, 콘솔에 원본/축소 크기를 함께 출력해 전처리 결과를 확인합니다.

```python
cv.imshow('Reduced Size - Color & Gray', combined)  # 결합 영상 출력
print(f"Original Size: {img.shape}")  # 원본 크기
print(f"Reduced Size: {img_small.shape}")  # 축소 크기
```

---

## 1) 마우스 이벤트로 좌/우 버튼 페인팅 처리

EVENT_LBUTTONDOWN, EVENT_RBUTTONDOWN, 그리고 드래그 상태(EVENT_MOUSEMOVE + 플래그)를 함께 처리해
한 번 클릭뿐 아니라 연속 붓질이 가능하도록 구성합니다.

```python
if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON)):
    cv.circle(img, (x, y), brush_size, L_color, -1)  # 파란색 붓질
elif event == cv.EVENT_RBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_RBUTTON)):
    cv.circle(img, (x, y), brush_size, R_color, -1)  # 빨간색 붓질
```

## 2) 키 입력으로 붓 크기 동적 조절

`cv.waitKey(1)` 루프에서 `+`, `-`, `q`를 실시간으로 받습니다.
min/max를 이용해 붓 크기가 지정 범위를 벗어나지 않도록 제한합니다.

```python
if key == ord('q'):
    break  # 종료
elif key == ord('+'):
    brush_size = min(brush_size + 1, 15)  # 최대 15
elif key == ord('-'):
    brush_size = max(brush_size - 1, 1)  # 최소 1
```

## 3) 콜백 등록 후 인터랙티브 창 유지

윈도우에 콜백을 연결해야 마우스 동작이 실제로 그림 작업으로 반영됩니다.

```python
cv.namedWindow('Painting')
cv.imshow('Painting', img)
cv.setMouseCallback('Painting', draw)  # 마우스 입력을 draw 함수에 연결
```

---

## 1) 드래그 기반 사각형 선택과 ROI 추출

마우스 다운에서 시작점을 기록하고, 업 이벤트에서 끝점을 받아 ROI를 확정합니다.
드래그 방향이 어떤 경우든 정상 처리되도록 min/max로 좌표를 정규화합니다.

```python
x1, x2 = min(ix, x), max(ix, x)
y1, y2 = min(iy, y), max(iy, y)

if x1 != x2 and y1 != y2:
    roi = ori_img[y1:y2, x1:x2]  # 원본에서 ROI 슬라이싱
    cv.imshow('Cropped ROI', roi)  # 선택 영역 미리보기
```

## 2) 드래그 중 실시간 시각화

사용자가 현재 어떤 영역을 선택 중인지 확인할 수 있도록,
마우스 이동마다 임시 복사본 위에 사각형을 그려 미리보기를 제공합니다.

```python
if drawing:
    img_draw = img.copy()  # 원본 작업 이미지 훼손 방지
    cv.rectangle(img_draw, (ix, iy), (x, y), (0, 255, 0), 2)
    cv.imshow('Select ROI', img_draw)
```

## 3) 키보드로 리셋/저장/종료 제어

`r`은 ROI 선택 상태 초기화, `s`는 ROI 파일 저장, `q`는 프로그램 종료입니다.
ROI가 없는 상태에서 저장을 누를 때도 안전하게 예외 메시지를 출력합니다.

```python
if key == ord('q'):
    break
elif key == ord('r'):
    img = ori_img.copy()
    roi = None
elif key == ord('s'):
    if roi is not None:
        cv.imwrite('soccer_roi.jpg', roi)
```

---

## 1) 체크보드 코너 검출과 정밀화

체크보드 내부 코너를 검출하고, cornerSubPix로 코너 위치를 서브픽셀 단위로 보정합니다.
이 좌표쌍이 캘리브레이션의 핵심 입력 데이터가 됩니다.

```python
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)  # 초기 코너 검출
if ret == True:
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)  # 정밀화
    objpoints.append(objp)      # 3D 실제 좌표
    imgpoints.append(corners2)  # 2D 이미지 좌표
```

## 2) 카메라 행렬과 왜곡 계수 계산

3D-2D 대응점 리스트를 이용해 카메라 내부 파라미터를 추정합니다.
여기서 계산된 K, dist는 이후 왜곡 보정 단계에서 사용됩니다.

```python
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)  # 내부 행렬 K, 왜곡 계수 dist 계산
```

## 3) 왜곡 보정 후 원본과 비교 시각화

undistort 결과를 원본과 나란히 붙여 육안으로 보정 효과를 확인합니다.

```python
dst = cv2.undistort(test_img, K, dist, None, new_camera_mtx)  # 왜곡 보정
result = np.hstack((test_img, dst))  # 원본/보정 결과 비교
cv2.imshow('Original vs Undistorted', result)
```

---

## 1) 회전 + 축소 행렬 생성

중심점 기준 회전과 스케일을 동시에 적용할 2x3 행렬을 생성합니다.

```python
center = (w // 2, h // 2)  # 중심 좌표
M = cv2.getRotationMatrix2D(center, 30, 0.8)  # 30도 회전 + 0.8배 축소
```

## 2) 평행이동을 같은 행렬에 직접 합성

M[0,2], M[1,2]를 직접 수정하면 별도 변환 없이 한 번에 이동까지 반영할 수 있습니다.

```python
M[0, 2] += 80   # x축 +80 이동
M[1, 2] += -40  # y축 -40 이동
```

## 3) warpAffine 적용 및 결과 저장

최종 변환 행렬로 이미지를 변환하고 결과를 파일로 남깁니다.

```python
dst = cv2.warpAffine(img, M, (w, h))  # 단일 아핀 변환 적용
cv2.imwrite('2W/transformed_rose.png', dst)  # 결과 저장
```

---

## 1) StereoBM으로 disparity 계산

좌/우 그레이스케일 이미지에서 시차를 계산하고, OpenCV 스케일(16배)을 원래 단위로 복원합니다.

```python
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)  # StereoBM 설정
disparity_raw = stereo.compute(left_gray, right_gray)          # 시차 계산
disparity = disparity_raw.astype(np.float32) / 16.0            # 실제 disparity로 변환
```

## 2) 깊이 계산과 ROI 평균 거리 분석

공식 $Z = \frac{fB}{d}$를 적용해 depth map을 만든 뒤, ROI별 평균 disparity/depth를 계산합니다.

```python
valid_mask = disparity > 0
depth_map = np.zeros(disparity.shape, dtype=np.float32)
depth_map[valid_mask] = (f * B) / disparity[valid_mask]  # 깊이 계산

for name, (x, y, w, h) in rois.items():
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
```

## 3) disparity/depth 컬러맵 생성과 시각화

시차/깊이를 각각 정규화해 컬러맵으로 변환하고, ROI 박스 이미지를 함께 출력/저장합니다.

```python
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_color.png"), depth_color)
cv2.imwrite(str(output_dir / "roi_left.png"), left_vis)
cv2.imwrite(str(output_dir / "roi_right.png"), right_vis)
```

---

## 1) cv.imread와 cvtColor로 입력을 준비

입력 이미지가 정상 로드되면 컬러(BGR) 이미지를 그레이스케일로 바꿉니다.
Sobel, Canny 같은 에지 연산은 밝기 변화 기반이므로 흑백 입력이 일반적으로 더 안정적입니다.

```python
original_bgr = cv.imread(image_path)  # 파일에서 원본 BGR 이미지 읽기
gray = cv.cvtColor(original_bgr, cv.COLOR_BGR2GRAY)  # 에지 계산용 흑백 영상 생성
```

## 2) Sobel x, y 성분 계산 후 magnitude 계산

Sobel x는 수직 경계(좌우 밝기 변화), Sobel y는 수평 경계(상하 밝기 변화)에 민감합니다.
두 성분을 합성해 최종 에지 강도 이미지를 만들고, 화면 표시를 위해 uint8로 변환합니다.

```python
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)  # x 방향 미분
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)  # y 방향 미분
edge_magnitude = cv.magnitude(sobel_x, sobel_y)  # 두 축 에지를 합쳐 전체 강도 계산
edge_uint8 = cv.convertScaleAbs(edge_magnitude)  # 시각화를 위해 0~255 범위로 변환
```

## 3) 원본과 에지 이미지를 나란히 시각화

왼쪽에는 원본, 오른쪽에는 Sobel 에지 강도를 배치하여 처리 전후를 한 번에 비교합니다.

```python
plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째(원본)
plt.imshow(original_rgb)
plt.title("Original Image")

plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째(에지)
plt.imshow(edge_uint8, cmap="gray")  # 강도 영상을 흑백 컬러맵으로 표시
plt.title("Sobel Edge Magnitude")
```

---

## 1) Canny로 에지 맵 생성

먼저 그레이스케일로 변환한 뒤, Canny의 두 임계값(100, 200)으로 강한 경계를 선별합니다.
이 단계의 결과가 이후 Hough 직선 검출의 입력이 됩니다.

```python
gray = cv.cvtColor(original_bgr, cv.COLOR_BGR2GRAY)  # Canny 전처리용 흑백 변환
edges = cv.Canny(gray, 100, 200)  # threshold1=100, threshold2=200
```

## 2) HoughLinesP로 직선 검출

확률적 허프 변환을 이용해 에지 픽셀 집합에서 선분을 찾습니다.
threshold, minLineLength, maxLineGap은 검출 민감도와 연결 품질을 결정하는 핵심 파라미터입니다.

```python
lines = cv.HoughLinesP(
	edges,  # Canny 에지 맵 입력
	rho=1,  # 거리 해상도(픽셀)
	theta=np.pi / 180,  # 각도 해상도(1도)
	threshold=80,  # 최소 누적 투표 수
	minLineLength=50,  # 최소 선분 길이
	maxLineGap=10,  # 끊긴 선분을 연결할 최대 간격
)
```

## 3) 직선을 빨간색으로 그려 시각화

검출된 각 선분의 시작점/끝점을 꺼내 원본 복사본 위에 빨간색으로 덧그립니다.
선 두께는 2로 지정해 가시성을 확보합니다.

```python
if lines is not None:
	for line in lines:
		x1, y1, x2, y2 = line[0]  # 검출된 선분 좌표 추출
		cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색 선 그리기
```

---

## 1) GrabCut 초기화와 사각형 설정

GrabCut은 마스크와 배경/전경 모델 배열을 기반으로 동작합니다.
초기 사각형(rect)은 객체가 포함될 가능성이 높은 영역을 지정해 분할의 시작점을 제공합니다.

```python
mask = np.zeros(original_bgr.shape[:2], np.uint8)  # 픽셀 라벨 저장용 마스크
bgd_model = np.zeros((1, 65), np.float64)  # 배경 GMM 모델 파라미터
fgd_model = np.zeros((1, 65), np.float64)  # 전경 GMM 모델 파라미터

h, w = original_bgr.shape[:2]
rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))  # (x, y, width, height)
```

## 2) GrabCut 수행 후 이진 마스크 변환

GrabCut 실행 후 마스크에는 확정/추정 배경·전경 라벨이 저장됩니다.
여기서는 배경 계열(GC_BGD, GC_PR_BGD)은 0, 전경 계열은 1로 변환해 후처리에 사용합니다.

```python
cv.grabCut(
	original_bgr,
	mask,
	rect,
	bgd_model,
	fgd_model,
	iterCount=5,
	mode=cv.GC_INIT_WITH_RECT,  # 사각형 기반 초기화 모드
)

mask_binary = np.where(
	(mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),  # 배경 또는 배경 추정
	0,  # 배경
	1,  # 전경
).astype("uint8")
```

## 3) 마스크로 배경 제거 및 3장 시각화

이진 마스크를 채널 방향으로 확장해 원본에 곱하면 배경이 제거된 전경만 남습니다.
원본, 마스크, 배경 제거 결과를 나란히 배치해 분할 품질을 빠르게 확인할 수 있습니다.

```python
foreground_bgr = original_bgr * mask_binary[:, :, np.newaxis]  # 배경 픽셀을 0으로 제거

plt.subplot(1, 3, 1)  # 원본
plt.imshow(original_rgb)
plt.title("Original Image")

plt.subplot(1, 3, 2)  # 마스크
plt.imshow(mask_binary, cmap="gray")  # 0=배경(검정), 1=전경(흰색)
plt.title("GrabCut Mask")

plt.subplot(1, 3, 3)  # 배경 제거 결과
plt.imshow(foreground_rgb)
plt.title("Foreground Only")
```

---

## 1) SIFT 생성 및 특징점 검출

SIFT 객체를 만들고 detectAndCompute를 통해 keypoint와 descriptor를 동시에 계산합니다.
매개변수(nfeatures, contrastThreshold, edgeThreshold, sigma)를 조정하면 검출 개수와 안정성이 달라집니다.

```python
sift = cv.SIFT_create(
    nfeatures=300,
    contrastThreshold=0.04,
    edgeThreshold=10,
    sigma=1.6,
)
keypoints, descriptors = sift.detectAndCompute(original_bgr, None)
```

## 2) drawKeypoints로 특징점 시각화

DRAW_RICH_KEYPOINTS 플래그를 사용해 점의 위치뿐 아니라 스케일/방향 정보까지 함께 표시합니다.
색상을 하나로 통일해 결과를 눈에 잘 띄게 했습니다.

```python
keypoint_viz_bgr = cv.drawKeypoints(
    original_bgr,
    keypoints,
    None,
    color=(0, 255, 0),
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)
```

## 3) matplotlib으로 원본/결과 비교 출력

원본과 특징점 시각화 이미지를 1행 2열로 배치해 비교합니다.
OpenCV(BGR)와 matplotlib(RGB)의 채널 순서 차이 때문에 표시 전 변환이 필요합니다.

```python
original_rgb = cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB)
keypoint_viz_rgb = cv.cvtColor(keypoint_viz_bgr, cv.COLOR_BGR2RGB)

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)

plt.subplot(1, 2, 2)
plt.imshow(keypoint_viz_rgb)
```

---

## 1) BFMatcher + knnMatch로 후보 매칭 생성

SIFT는 float descriptor를 쓰므로 BFMatcher에서 L2 거리 기준이 일반적입니다.
knnMatch(k=2)로 각 특징점마다 최근접/차근접 후보를 받아 ratio test를 적용할 준비를 합니다.

```python
matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
```

## 2) Lowe Ratio Test로 좋은 매칭 필터링

최근접 매칭이 차근접 매칭보다 충분히 가까울 때만 채택해 오매칭을 줄입니다.
이 코드에서는 임계값 0.75를 사용합니다.

```python
ratio_threshold = 0.75
good_matches = []
for pair in knn_matches:
    if len(pair) < 2:
        continue
    m, n = pair
    if m.distance < ratio_threshold * n.distance:
        good_matches.append(m)
```

## 3) drawMatches로 매칭 결과 시각화

정렬된 좋은 매칭 중 상위 일부만 표시해 화면 복잡도를 줄이고,
결과를 matplotlib으로 한 번에 확인합니다.

```python
good_matches = sorted(good_matches, key=lambda x: x.distance)
matches_to_draw = good_matches[:120]

match_vis_bgr = cv.drawMatches(
    image1_bgr,
    keypoints1,
    image2_bgr,
    keypoints2,
    matches_to_draw,
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)
```

---

## 1) 좋은 매칭점으로 호모그래피 대응점 구성

호모그래피는 두 영상의 대응점 쌍이 필요합니다.
여기서는 good_matches에서 img2(src)와 img1(dst) 좌표를 꺼내 findHomography 입력 형식으로 만듭니다.

```python
src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
```

## 2) RANSAC으로 호모그래피 추정

일부 오매칭(outlier)이 포함되어도 RANSAC을 사용하면 안정적인 변환 행렬을 구할 수 있습니다.
반환된 inlier_mask를 이용해 실제 정합에 기여한 매칭만 골라낼 수 있습니다.

```python
homography, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
```

## 3) warpPerspective 정렬 + 매칭 결과 동시 출력

출력 크기를 (w1+w2, max(h1,h2))로 잡아 파노라마 형태로 정렬 결과를 확인합니다.
동시에 drawMatches 결과를 옆에 배치해 정합 품질을 육안으로 점검합니다.

```python
panorama_width = w1 + w2
panorama_height = max(h1, h2)
warped_bgr = cv.warpPerspective(image2_bgr, homography, (panorama_width, panorama_height))
warped_bgr[0:h1, 0:w1] = image1_bgr
```

---

## 1) MNIST 로드 + 재분할

기본 제공 train/test를 그대로 쓰지 않고 전체를 합쳐 다시 셔플 분할해, 데이터 분할 과정을 코드로 명확히 확인할 수 있도록 구성했습니다.

```python
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()  # MNIST 원본 train/test를 불러옵니다.

all_images = np.concatenate([x_train_raw, x_test_raw], axis=0)  # 이미지 데이터를 하나로 합칩니다.
all_labels = np.concatenate([y_train_raw, y_test_raw], axis=0)  # 라벨 데이터도 같은 순서로 합칩니다.

x_train, x_test, y_train, y_test = split_dataset(all_images, all_labels, train_ratio=0.8, seed=42)  # 셔플 후 8:2로 재분할합니다.
```

## 2) MLP 모델 구성 및 학습

Flatten으로 입력을 펼친 뒤 Dense 2개 은닉층을 거쳐 10클래스 softmax를 출력합니다.

```python
model = models.Sequential(
	[
		layers.Flatten(input_shape=(28, 28)),  # 28x28 이미지를 1차원 벡터로 펼칩니다.
		layers.Dense(128, activation="relu"),  # 첫 번째 은닉층입니다.
		layers.Dense(64, activation="relu"),  # 두 번째 은닉층입니다.
		layers.Dense(10, activation="softmax"),  # 10개 숫자 클래스 확률을 출력합니다.
	]
)
```

## 3) 테스트 평가 + 혼동행렬

단일 정확도뿐 아니라 혼동행렬을 함께 출력해 어떤 클래스에서 오분류가 발생하는지 확인할 수 있도록 했습니다.

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)  # 테스트셋 손실과 정확도를 계산합니다.
probabilities = model.predict(x_test, verbose=0)  # 각 샘플의 클래스별 확률을 예측합니다.
predictions = np.argmax(probabilities, axis=1)  # 가장 높은 확률의 클래스를 최종 예측값으로 선택합니다.
confusion = tf.math.confusion_matrix(y_test, predictions, num_classes=10).numpy()  # 혼동행렬을 계산합니다.
```

---

## 1) 데이터 전처리와 증강

기본 정규화(0~1) 이후 학습 시점에 랜덤 증강을 적용해 일반화 성능을 높였습니다.

```python
def preprocess_dataset(x_train, x_test):
	x_train = x_train.astype("float32") / 255.0  # 훈련 이미지 픽셀값을 0~1로 정규화합니다.
	x_test = x_test.astype("float32") / 255.0  # 테스트 이미지 픽셀값도 같은 방식으로 정규화합니다.
	return x_train, x_test  # 정규화된 훈련/테스트 데이터를 반환합니다.

def build_data_augmentation():
	return models.Sequential(
		[
			layers.RandomFlip("horizontal"),  # 좌우 반전으로 시점 다양성을 늘립니다.
			layers.RandomRotation(0.08),  # 작은 회전 변형을 주어 일반화 성능을 높입니다.
			layers.RandomZoom(height_factor=0.1, width_factor=0.1),  # 확대/축소 변형으로 크기 변화에 대응합니다.
			layers.RandomContrast(factor=0.1),  # 대비 변형으로 조명 변화에 대응합니다.
		]
	)
```

## 2) CNN 모델 및 학습 안정화

Conv 블록 뒤 BatchNormalization을 배치하고, EarlyStopping + ReduceLROnPlateau를 함께 사용해 과적합과 학습 정체를 완화했습니다.

```python
layers.Conv2D(32, (3, 3), activation="relu", padding="same"),  # 특징맵을 추출하는 합성곱 층입니다.
layers.BatchNormalization(),  # 배치 정규화로 학습을 안정화합니다.
layers.MaxPooling2D((2, 2)),  # 공간 크기를 줄여 연산량과 과적합을 완화합니다.

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)  # 검증 정확도 개선이 멈추면 조기 종료합니다.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)  # 정체 시 학습률을 절반으로 낮춥니다.
```

## 3) dog.jpg 예측 안정화(TTA)

단일 리사이즈 1회 예측 대신, 5개 크롭 + 좌우 반전(총 10뷰) 예측 평균을 사용해 cat/dog 혼동을 줄였습니다.

```python
square_views = make_square_crop_views(image_rgb)  # 중심/코너 기준 정사각 크롭 5개를 생성합니다.

batch = []
for view in square_views:
	resized = cv.resize(view, (32, 32), interpolation=cv.INTER_AREA)  # CIFAR-10 입력 크기(32x32)로 맞춥니다.
	batch.append(resized)  # 원본 크롭 뷰를 추가합니다.
	batch.append(cv.flip(resized, 1))  # 좌우 반전 뷰를 추가해 예측을 보강합니다.

probs_batch = model.predict(batch_input, verbose=0)  # 여러 뷰 각각의 클래스 확률을 예측합니다.
probs = np.mean(probs_batch, axis=0)  # 뷰별 확률을 평균내 최종 확률로 사용합니다.
```

---

## 1) YOLOv3 검출 + SORT 추적

YOLOv3로 매 프레임의 객체를 검출하고, SORT가 그 결과를 프레임 간에 연결해 동일 객체에 같은 ID를 부여합니다.

```python
net = load_darknet_net(args.cfg, args.weights)  # 한글 경로 문제를 포함해 YOLO 모델을 안전하게 로드합니다.
detections, class_ids, confidences = detect_objects_yolo(
	net,
	frame,
	conf_threshold=args.conf_thres,
	nms_threshold=args.nms_thres,
	input_size=args.input_size,
)

tracks = tracker.update(detections)  # 검출 결과를 SORT 추적기에 넣어 추적 상태를 갱신합니다.
```

## 2) Kalman Filter + IoU 연관

SORT는 칼만 필터로 다음 위치를 예측하고, IoU가 가장 큰 검출 박스를 헝가리안 알고리즘으로 매칭합니다.

```python
if SCIPY_AVAILABLE:
	row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # IoU를 최대화하는 매칭을 찾습니다.
else:
	matched_indices = greedy_assignment(iou_matrix)  # scipy가 없으면 greedy 방식으로 대체합니다.
```

## 3) 결과 시각화

추적된 객체는 ID와 클래스 이름을 함께 표시해 실시간으로 확인할 수 있습니다.

```python
cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 객체 경계 상자를 그립니다.
cv.putText(frame, label, (x1, max(20, y1 - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # ID와 클래스명을 표시합니다.
```

---

## 1) FaceMesh 초기화와 웹캠 캡처

MediaPipe의 FaceMesh를 사용해 얼굴 랜드마크 검출기를 만들고, OpenCV로 웹캠 영상을 받아옵니다.

```python
mp_face_mesh = mp.solutions.face_mesh  # FaceMesh 모듈을 준비합니다.
cap = cv.VideoCapture(0)  # 웹캠을 엽니다.
```

## 2) 랜드마크 좌표를 픽셀로 변환

랜드마크는 0~1 범위의 정규화 좌표이므로, 프레임의 너비와 높이를 곱해 실제 픽셀 좌표로 바꿔야 합니다.

```python
x = int(lm.x * w)  # x 좌표를 픽셀 좌표로 변환합니다.
y = int(lm.y * h)  # y 좌표를 픽셀 좌표로 변환합니다.
cv.circle(frame_bgr, (x, y), 1, (0, 255, 0), -1)  # 각 랜드마크를 점으로 그립니다.
```

## 3) ESC 종료 처리

실시간 카메라 확인 중 ESC 키를 누르면 프로그램이 종료되도록 구성했습니다.

```python
key = cv.waitKey(1) & 0xFF  # 키 입력을 확인합니다.
if key == 27:
	break  # ESC 키를 누르면 종료합니다.
```
