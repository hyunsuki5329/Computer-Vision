import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# 이미지 로딩을 담당하는 보조 함수 정의
def load_bgr_image(image_path: str):
    """Load image robustly, including non-ASCII paths on Windows."""
    # 1차 시도: 일반적인 OpenCV 이미지 로드
    image_bgr = cv.imread(image_path)
    # 1차 로드 실패 시 보조 로딩 경로로 진입
    if image_bgr is None:
        # 파일 바이트를 uint8 배열로 직접 읽기
        raw = np.fromfile(image_path, dtype=np.uint8)
        # 읽은 바이트가 비어 있지 않으면 디코딩 시도
        if raw.size > 0:
            # 바이트 배열을 컬러 이미지로 디코딩
            image_bgr = cv.imdecode(raw, cv.IMREAD_COLOR)
    # 최종 로드 결과 반환 (성공 시 ndarray, 실패 시 None)
    return image_bgr


def main():
    # 1) 입력 이미지 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "mot_color70.jpg")

    # 2) 이미지 로드
    # 보조 함수로 BGR 이미지 로딩 수행
    original_bgr = load_bgr_image(image_path)
    if original_bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    # 3) SIFT 객체 생성 (매개변수 조정 가능)
    # SIFT 검출기/기술자 객체 생성
    sift = cv.SIFT_create(
        # 최대 특징점 개수를 300개로 제한
        nfeatures=300,
        # 대비 임계값 설정 (값이 작을수록 더 많은 특징점)
        contrastThreshold=0.04,
        # 에지 응답 제거 임계값 설정
        edgeThreshold=10,
        # 가우시안 스케일 공간의 초기 시그마 설정
        sigma=1.6,
    )

    # 4) 특징점 검출 및 기술자 계산
    # 입력 이미지에서 keypoints와 descriptors를 동시에 계산
    keypoints, descriptors = sift.detectAndCompute(original_bgr, None)

    # 5) 특징점 시각화 (크기/방향 포함)
    # 원본 이미지 위에 특징점을 덧그린 시각화 이미지 생성
    keypoint_viz_bgr = cv.drawKeypoints(
        # 입력 원본 BGR 이미지
        original_bgr,
        # 검출된 특징점 목록
        keypoints,
        # 출력 이미지 버퍼를 OpenCV가 새로 생성하도록 None 전달
        None,
        # 특징점 색상을 초록색(BGR)으로 통일
        color=(0, 255, 0),
        # 원형 크기/방향까지 보이는 rich keypoint 플래그 사용
        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    # 6) matplotlib 표시를 위한 BGR -> RGB 변환
    # 원본 이미지를 matplotlib 표시에 맞게 RGB로 변환
    original_rgb = cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB)
    # 특징점 시각화 이미지를 matplotlib 표시에 맞게 RGB로 변환
    keypoint_viz_rgb = cv.cvtColor(keypoint_viz_bgr, cv.COLOR_BGR2RGB)

    # 7) 결과 출력
    plt.figure(figsize=(14, 6))

    # 1행 2열 중 첫 번째 서브플롯 선택
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # 1행 2열 중 두 번째 서브플롯 선택
    plt.subplot(1, 2, 2)
    plt.imshow(keypoint_viz_rgb)
    plt.title(f"SIFT Keypoints (count={len(keypoints)})")
    plt.axis("off")

    # 서브플롯 간 간격 자동 조정
    plt.tight_layout()
    plt.show()

    # 콘솔 정보 출력
    if descriptors is None:
        print(f"Detected keypoints: {len(keypoints)}")
        print("Descriptors: None")
    else:
        print(f"Detected keypoints: {len(keypoints)}")
        print(f"Descriptors shape: {descriptors.shape}")


if __name__ == "__main__":
    main()
