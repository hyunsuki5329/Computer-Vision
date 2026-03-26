import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def load_bgr_image(image_path: str):
    image_bgr = cv.imread(image_path)
    if image_bgr is None:
        # 파일 바이트를 uint8 배열로 직접 읽기
        raw = np.fromfile(image_path, dtype=np.uint8)
        # 읽은 바이트가 비어 있지 않으면 디코딩 시도
        if raw.size > 0:
            # 바이트 배열을 컬러 이미지로 디코딩
            image_bgr = cv.imdecode(raw, cv.IMREAD_COLOR)
    # 최종 로드 결과 반환 (성공 시 ndarray, 실패 시 None)
    return image_bgr


def save_bgr_image(image_path: str, image_bgr):
    ext = os.path.splitext(image_path)[1] or ".png"
    success, encoded = cv.imencode(ext, image_bgr)
    if not success:
        raise RuntimeError(f"이미지 인코딩에 실패했습니다: {image_path}")
    encoded.tofile(image_path)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image1_path = os.path.join(script_dir, "mot_color70.jpg")
    image2_path = os.path.join(script_dir, "mot_color80.jpg")

    # 두 이미지를 BGR 형식으로 로드
    image1_bgr = load_bgr_image(image1_path)
    image2_bgr = load_bgr_image(image2_path)
    if image1_bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image1_path}")
    if image2_bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image2_path}")

    # SIFT 계산 전, 두 이미지를 그레이스케일로 변환
    gray1 = cv.cvtColor(image1_bgr, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2_bgr, cv.COLOR_BGR2GRAY)

    # SIFT 검출기/기술자 객체 생성
    # nfeatures: 추출할 특징점 최대 개수(너무 많으면 매칭/시각화가 복잡해짐)
    # contrastThreshold: 값이 낮을수록 약한 코너까지 더 많이 검출
    # edgeThreshold: 에지(선분)처럼 불안정한 특징 제거 강도 조절
    # sigma: 스케일 공간의 초기 가우시안 블러 강도
    sift = cv.SIFT_create(nfeatures=600, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

    # 두 이미지에서 특징점과 디스크립터를 각각 계산
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # 디스크립터 계산 실패 시 후속 매칭이 불가능하므로 예외 처리
    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("특징점 디스크립터를 생성하지 못했습니다.")

    # L2 거리 기반 BFMatcher 생성 (SIFT에 적합)
    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    # 각 특징점에 대해 최근접 2개 이웃을 찾는 KNN 매칭 수행
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # KNN 원시 매칭 시각화를 위한 2개 이웃 쌍 필터링
    knn_pairs = [pair for pair in knn_matches if len(pair) == 2]
    knn_pairs_to_draw = knn_pairs[:120]
    knn_raw_bgr = cv.drawMatchesKnn(
        image1_bgr,
        keypoints1,
        image2_bgr,
        keypoints2,
        knn_pairs_to_draw,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Lowe ratio test 임계값 설정
    ratio_threshold = 0.75
    # ratio test를 통과한 좋은 매칭만 저장할 리스트
    good_matches = []
    # KNN 결과를 순회하며 비율 테스트 적용
    for pair in knn_matches:
        # 이웃이 2개 미만인 경우 비율 테스트 불가
        if len(pair) < 2:
            continue
        # 가장 가까운 매칭과 두 번째 가까운 매칭 분리
        m, n = pair
        # 최근접 거리가 충분히 작을 때만 좋은 매칭으로 채택
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # 매칭 품질이 좋은 순(거리 오름차순)으로 정렬
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    # 시각화 복잡도를 줄이기 위해 상위 120개만 표시
    matches_to_draw = good_matches[:120]

    # 두 이미지와 매칭선을 하나의 결과 이미지로 렌더링
    match_vis_bgr = cv.drawMatches(
        # 왼쪽 이미지
        image1_bgr,
        # 왼쪽 이미지 특징점
        keypoints1,
        # 오른쪽 이미지
        image2_bgr,
        # 오른쪽 이미지 특징점
        keypoints2,
        # 화면에 그릴 매칭 목록
        matches_to_draw,
        # 출력 이미지는 OpenCV가 새로 생성
        None,
        # 매칭되지 않은 단일 특징점은 생략
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # matplotlib 표시를 위해 BGR -> RGB 변환
    match_vis_rgb = cv.cvtColor(match_vis_bgr, cv.COLOR_BGR2RGB)

    # README용 중간/최종 결과 이미지 저장
    input_pair_bgr = cv.hconcat([image1_bgr, image2_bgr])
    save_bgr_image(os.path.join(script_dir, "result2_input_pair.png"), input_pair_bgr)
    save_bgr_image(os.path.join(script_dir, "result2_knn_raw.png"), knn_raw_bgr)
    save_bgr_image(os.path.join(script_dir, "result2.png"), match_vis_bgr)

    # 결과 출력용 Figure 생성
    plt.figure(figsize=(16, 7))
    plt.imshow(match_vis_rgb)
    plt.title(
        f"SIFT Matching | total={len(knn_matches)} | good={len(good_matches)} | drawn={len(matches_to_draw)}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"Image1 keypoints: {len(keypoints1)}")
    print(f"Image2 keypoints: {len(keypoints2)}")
    print(f"KNN matches: {len(knn_matches)}")
    print(f"Good matches (ratio test): {len(good_matches)}")


if __name__ == "__main__":
    main()
