import cv2
import numpy as np
from pathlib import Path


# 출력 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기
left_color = cv2.imread("2W/left.png")
right_color = cv2.imread("2W/right.png")

# 카메라 파라미터
f = 700.0  # 초점 거리 (focal length)
B = 0.12   # 베이스라인 (두 카메라 사이 거리, 12cm)

# ROI 설정 (x, y, w, h)
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# [그레이스케일 변환] - 스테레오 매칭을 위해 흑백 변환 필수
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------
# 1. Disparity(시차) 계산
# ---------------------------------------------------------
# cv2.StereoBM_create: 블록 매칭(Block Matching) 알고리즘 객체를 생성합니다.
# numDisparities: 왼쪽-오른쪽 이미지 사이에서 탐색할 최대 픽셀 거리 차이입니다. 반드시 16의 배수여야 합니다.
# blockSize: 매칭 시 비교할 작은 사각형 영역의 크기
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# .compute: 왼쪽과 오른쪽 흑백 영상을 비교하여 시차 지도(Disparity Map)를 생성합니다.
disparity_raw = stereo.compute(left_gray, right_gray)

# [중요] StereoBM 알고리즘은 연산의 정밀도를 위해 내부적으로 결과값에 16을 곱하여 정수형으로 반환합니다.
# 실제 픽셀 단위의 시차값(d)을 얻으려면 반드시 float32 타입으로 바꾼 뒤 16으로 나누어야 합니다.
disparity = disparity_raw.astype(np.float32) / 16.0



# ---------------------------------------------------------
# 2. Depth(깊이/거리) 계산 (Z = fB / d)
# ---------------------------------------------------------
# 시차(disparity) 정보를 실제 세계의 거리 단위(m)인 Depth로 변환하는 과정입니다.
# 결과를 담을 빈 배열을 원본 이미지와 같은 크기로 생성합니다.
depth_map = np.zeros(disparity.shape, dtype=np.float32)

# 시차(d)가 0인 픽셀은 '무한히 먼 곳'이거나 '매칭 실패' 영역이므로 제외해야 합니다.
# (0으로 나누기 에러 방지 및 유효한 데이터만 추출)
valid_mask = disparity > 0

# 공식 적용: Z (거리) = f (초점거리) * B (카메라 간격) / d (시차)
# d가 분모에 있으므로, 시차(d)가 클수록 거리(Z)는 짧아집니다(가까워집니다).
depth_map[valid_mask] = (f * B) / disparity[valid_mask]



# ---------------------------------------------------------
# 3. ROI(관심 영역)별 평균 disparity / depth 계산
# ---------------------------------------------------------
results = {}

# 딕셔너리에 저장된 각 객체(Painting, Frog, Teddy)의 좌표를 순회합니다.
for name, (x, y, w, h) in rois.items():
    # 이미지 슬라이싱을 통해 해당 물체가 위치한 영역(Box)만 잘라냅니다.
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    
    # 해당 사각형 영역 안에서도 시차가 유효한(>0) 픽셀들만 골라내어 마스크를 만듭니다.
    valid_roi_mask = roi_disp > 0
    
    # 영역 내의 유효한 값들의 평균을 계산합니다. (값이 없으면 0으로 처리)
    # 평균 시차(avg_disp)와 평균 거리(avg_depth)를 구합니다.
    avg_disp = np.mean(roi_disp[valid_roi_mask]) if np.any(valid_roi_mask) else 0
    avg_depth = np.mean(roi_depth[valid_roi_mask]) if np.any(valid_roi_mask) else 0
    
    # 결과를 사물 이름별로 저장합니다.
    results[name] = (avg_disp, avg_depth)

# ---------------------------------------------------------
# 4. 결과 출력 및 데이터 해석
# ---------------------------------------------------------
# f-string을 이용해 표 형태로 깔끔하게 출력합니다.
print(f"{'Object':<10} | {'Avg Disparity':<15} | {'Avg Depth (m)':<15}")
print("-" * 45)
for name, (d, z) in results.items():
    print(f"{name:<10} | {d:<15.2f} | {z:<15.4f}")

# min/max 함수와 key 인자를 활용해 결과 분석
# 가장 가까운 물체: Depth(z) 값이 가장 작은 객체를 찾습니다.
closest = min(results, key=lambda k: results[k][1])

# 가장 먼 물체: Depth(z) 값이 가장 큰 객체를 찾습니다.
farthest = max(results, key=lambda k: results[k][1])

print(f"\n해석: 가장 가까운 물체는 '{closest}', 가장 먼 물체는 '{farthest}'입니다.")

# ---------------------------------------------------------
# 5. Disparity(시차) 시각화
# 목적: 0~64 범위의 시차 값을 0~255 색상으로 변환하여 열지도(Heatmap) 생성
# ---------------------------------------------------------
# 데이터 오염을 방지하기 위해 복사본 생성 후, 값이 없는(<=0) 부분은 NaN(계산 제외) 처리
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

# 모든 픽셀이 매칭에 실패했을 경우를 대비한 예외 처리
if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

# 이상치(Outlier)를 제거하고 색상 대비를 높이기 위해 하위 5%, 상위 95% 값을 기준으로 설정
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

# 0으로 나누기 방지를 위한 안전 장치
if d_max <= d_min:
    d_max = d_min + 1e-6

# 데이터를 0.0 ~ 1.0 사이로 정규화 (Normalization)
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

# 0~255 범위의 8비트 정수형 이미지로 변환
disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# COLORMAP_JET 적용: 큰 값(가까운 곳)은 빨간색, 작은 값(먼 곳)은 파란색으로 표시
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)



# ---------------------------------------------------------
# 6. Depth(깊이) 시각화
# 목적: 실제 거리(m) 데이터를 색상으로 표현 (가까울수록 빨강, 멀수록 파랑)
# ---------------------------------------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    # 거리 데이터도 마찬가지로 하위 5%, 상위 95%를 기준으로 정규화 범위 설정
    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    # 0.0(가장 가까움) ~ 1.0(가장 멂)으로 변환
    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # [중요] Depth는 값이 클수록 멀기 때문에, 1.0에서 빼주어 반전시킵니다.
    # 이렇게 해야 시각적으로 '가까운 물체가 빨간색'으로 통일됩니다.
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

# 최종적으로 색상을 입힘
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# ---------------------------------------------------------
# 7. Left / Right 이미지에 ROI(관심 영역) 표시
# 목적: 분석 대상인 Painting, Frog, Teddy가 어디인지 이미지에 사각형으로 그림
# ---------------------------------------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    # cv2.rectangle: 원본 이미지에 (x, y)부터 (x+w, y+h)까지 초록색(0, 255, 0) 사각형 그림
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.putText: 사각형 위에 물체의 이름(Painting 등)을 텍스트로 표기
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 오른쪽 이미지에도 동일하게 표시하여 두 시점의 차이를 확인 가능하게 함
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "disparity_color.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_color.png"), depth_color)
cv2.imwrite(str(output_dir / "roi_left.png"), left_vis)
cv2.imwrite(str(output_dir / "roi_right.png"), right_vis)

# -----------------------------
# 9. 출력 (이미지 보기)
# -----------------------------
cv2.imshow("Disparity Map", disparity_color)
cv2.imshow("Depth Map", depth_color)
cv2.imshow("ROI Selection", left_vis)
cv2.imshow("ROI Selection (Right)", right_vis)

cv2.waitKey(0)
cv2.destroyAllWindows()