import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def load_bgr_image(image_path: str):
    image_bgr = cv.imread(image_path)
    if image_bgr is None:
        raw = np.fromfile(image_path, dtype=np.uint8)
        if raw.size > 0:
            image_bgr = cv.imdecode(raw, cv.IMREAD_COLOR)
    return image_bgr


def save_bgr_image(image_path: str, image_bgr):
    ext = os.path.splitext(image_path)[1] or ".png"
    success, encoded = cv.imencode(ext, image_bgr)
    if not success:
        raise RuntimeError(f"이미지 인코딩에 실패했습니다: {image_path}")
    encoded.tofile(image_path)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 샘플 이미지 2장 선택 (img1, img2)
    image1_path = os.path.join(script_dir, "img1.jpg")
    image2_path = os.path.join(script_dir, "img2.jpg")

    # 기준 이미지(img1)와 정렬 대상 이미지(img2) 로드
    image1_bgr = load_bgr_image(image1_path)
    image2_bgr = load_bgr_image(image2_path)
    if image1_bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image1_path}")
    if image2_bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image2_path}")

    # SIFT 계산을 위해 그레이스케일로 변환
    gray1 = cv.cvtColor(image1_bgr, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2_bgr, cv.COLOR_BGR2GRAY)

    # SIFT 특징점 검출기 생성
    # nfeatures: 추출할 특징점 최대 개수
    # contrastThreshold: 값이 작을수록 약한 특징까지 검출
    # edgeThreshold: 에지성 특징 제거 강도 조절
    # sigma: 초기 가우시안 블러 강도
    sift = cv.SIFT_create(nfeatures=1000, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

    # 두 이미지에서 특징점과 디스크립터 계산
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # 디스크립터 계산 실패 시 후속 매칭이 불가능하므로 예외 처리
    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("특징점 디스크립터를 생성하지 못했습니다.")

    # BFMatcher + KNN(2-NN) 매칭
    # SIFT 디스크립터는 실수 벡터이므로 L2 거리(norm)를 사용
    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    # crossCheck를 끄고 k=2로 최근접 2개 이웃을 받아 ratio test에 사용
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Lowe ratio test로 좋은 매칭점만 선별
    ratio_threshold = 0.7
    # ratio test를 통과한 좋은 매칭만 저장할 리스트
    good_matches = []
    # KNN 결과를 순회하며 비율 테스트 적용
    for pair in knn_matches:
        # 이웃이 2개 미만이면 비율 테스트 불가
        if len(pair) < 2:
            continue
        # 가장 가까운 매칭과 두 번째 가까운 매칭 분리
        m, n = pair
        # 최근접 거리가 충분히 작을 때만 좋은 매칭으로 채택
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # 호모그래피 계산 최소 조건(4쌍) 미만이면 중단
    if len(good_matches) < 4:
        raise RuntimeError(f"호모그래피 계산에 필요한 매칭점이 부족합니다. good_matches={len(good_matches)}")

    # good matches 기준 시각화(호모그래피 전 단계 중간 결과)
    good_matches_to_draw = good_matches[:120]
    good_matches_bgr = cv.drawMatches(
        image1_bgr,
        keypoints1,
        image2_bgr,
        keypoints2,
        good_matches_to_draw,
        None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # 호모그래피 계산: img2 좌표를 img1 좌표계로 변환
    # trainIdx는 두 번째 이미지(keypoints2)에서의 매칭 인덱스
    # 즉 src_pts는 "변환할 원본"인 img2의 대응점 집합
    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # queryIdx는 첫 번째 이미지(keypoints1)에서의 매칭 인덱스
    # 즉 dst_pts는 "맞춰질 기준"인 img1의 대응점 집합
    # reshape(-1, 1, 2)는 findHomography가 요구하는 점 배열 형태(N,1,2)
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC 기반 호모그래피 계산 (이상점 영향 완화)
    homography, inlier_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    # 호모그래피 계산 실패 시 예외 처리
    if homography is None:
        raise RuntimeError("호모그래피 행렬 계산에 실패했습니다.")

    # 파노라마 크기: (w1+w2, max(h1, h2))
    h1, w1 = image1_bgr.shape[:2]
    h2, w2 = image2_bgr.shape[:2]
    panorama_width = w1 + w2
    panorama_height = max(h1, h2)

    # img2를 호모그래피로 변환하여 img1 기준으로 정렬
    warped_bgr = cv.warpPerspective(image2_bgr, homography, (panorama_width, panorama_height))
    # 기준 이미지(img1)를 좌측 상단에 배치해 기준면을 유지
    warped_bgr[0:h1, 0:w1] = image1_bgr

    # RANSAC inlier만 선택해 매칭 시각화 품질 개선
    inlier_matches = []
    # inlier 마스크가 있으면 inlier 매칭만 필터링
    if inlier_mask is not None:
        # (N,1) 형태 마스크를 1차원 리스트로 평탄화
        mask_flat = inlier_mask.ravel().tolist()
        # good_matches와 마스크를 짝지어 keep=1(inlier)인 매칭만 유지
        inlier_matches = [m for m, keep in zip(good_matches, mask_flat) if keep]
    else:
        # 마스크가 없으면 good matches 전체 사용
        inlier_matches = good_matches

    # 표시할 매칭 수를 제한해 시각적 복잡도 감소
    matches_to_draw = inlier_matches[:120]

    # 특징점 매칭 결과 시각화
    matching_result_bgr = cv.drawMatches(
        # 기준 이미지(img1)
        image1_bgr,
        # 기준 이미지 특징점
        keypoints1,
        # 정렬 대상 이미지(img2)
        image2_bgr,
        # 대상 이미지 특징점
        keypoints2,
        # 화면에 그릴 매칭 목록
        matches_to_draw,
        # 출력 이미지는 OpenCV가 새로 생성
        None,
        # 단일(비매칭) 특징점은 생략
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # README용 중간/최종 결과 이미지 저장
    save_bgr_image(os.path.join(script_dir, "result3_good_matches.png"), good_matches_bgr)
    save_bgr_image(os.path.join(script_dir, "result3_inlier_matches.png"), matching_result_bgr)
    save_bgr_image(os.path.join(script_dir, "result3_warped.png"), warped_bgr)

    final_panel_bgr = cv.hconcat([warped_bgr, matching_result_bgr])
    save_bgr_image(os.path.join(script_dir, "result3.png"), final_panel_bgr)

    # matplotlib 표시를 위해 결과 이미지를 BGR -> RGB로 변환
    warped_rgb = cv.cvtColor(warped_bgr, cv.COLOR_BGR2RGB)
    # 매칭 시각화 이미지도 동일하게 BGR -> RGB 변환
    matching_result_rgb = cv.cvtColor(matching_result_bgr, cv.COLOR_BGR2RGB)

    # 변환된 이미지와 매칭 결과를 나란히 출력
    plt.figure(figsize=(18, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(warped_rgb)
    plt.title("Warped Image (Image Alignment)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(matching_result_rgb)
    plt.title(
        f"Matching Result | KNN={len(knn_matches)} | Good={len(good_matches)} | Inlier={len(inlier_matches)}"
    )
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Image1 keypoints: {len(keypoints1)}")
    print(f"Image2 keypoints: {len(keypoints2)}")
    print(f"KNN matches: {len(knn_matches)}")
    print(f"Good matches (ratio<{ratio_threshold}): {len(good_matches)}")
    print(f"Inlier matches (RANSAC): {len(inlier_matches)}")


if __name__ == "__main__":
    main()
