import argparse  # 실행 인자 파싱을 위해 사용합니다.
import importlib  # scipy를 런타임에 선택적으로 불러오기 위해 사용합니다.
import os  # 파일 경로와 존재 여부를 확인하기 위해 사용합니다.
import shutil  # 모델 파일을 임시 ASCII 경로로 복사하기 위해 사용합니다.
import tempfile  # 임시 캐시 디렉터리를 만들기 위해 사용합니다.
import time  # FPS 계산용 시간을 측정하기 위해 사용합니다.

import cv2 as cv  # OpenCV로 영상 입출력, DNN, 시각화를 처리합니다.
import numpy as np  # 배열 연산과 IoU 계산을 위해 사용합니다.

try:
    scipy_optimize = importlib.import_module("scipy.optimize")  # scipy.optimize를 동적으로 불러옵니다.
    linear_sum_assignment = scipy_optimize.linear_sum_assignment  # 헝가리안 알고리즘 함수를 가져옵니다.
    SCIPY_AVAILABLE = True  # scipy 사용 가능 여부를 표시합니다.
except Exception:
    linear_sum_assignment = None  # scipy가 없으면 매칭 함수는 비워 둡니다.
    SCIPY_AVAILABLE = False  # scipy 미설치 상태로 표시합니다.


DEFAULT_COCO_CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_class_names(names_path: str):
    if os.path.exists(names_path):  # 클래스 이름 파일이 있으면 읽습니다.
        with open(names_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]  # 빈 줄을 제거하고 이름을 읽습니다.
        if names:
            return names  # 파일에서 읽은 클래스 목록을 반환합니다.
    return DEFAULT_COCO_CLASSES  # 없으면 내장 COCO 클래스 목록을 사용합니다.


def _make_ascii_cache_path(src_path: str):
    abs_path = os.path.abspath(src_path)  # 입력 경로를 절대 경로로 바꿉니다.
    cache_root = os.path.join(tempfile.gettempdir(), "cv_yolo_cache")  # 임시 캐시 폴더를 정합니다.
    os.makedirs(cache_root, exist_ok=True)  # 캐시 폴더가 없으면 생성합니다.

    file_hash = format(abs(hash(abs_path)), "x")  # 경로를 기준으로 간단한 해시를 만듭니다.
    _, ext = os.path.splitext(abs_path)  # 파일 확장자를 분리합니다.
    safe_name = f"model_{file_hash}{ext.lower()}"  # 충돌이 적은 ASCII 안전 이름을 만듭니다.
    return os.path.join(cache_root, safe_name)  # 최종 캐시 경로를 반환합니다.


def _ensure_ascii_copy(src_path: str):
    dst_path = _make_ascii_cache_path(src_path)  # 복사할 ASCII 경로를 계산합니다.
    need_copy = True  # 기본적으로 복사가 필요하다고 가정합니다.

    if os.path.exists(dst_path):  # 이미 캐시가 있으면 크기를 비교합니다.
        try:
            need_copy = os.path.getsize(dst_path) != os.path.getsize(src_path)  # 내용이 다를 때만 복사합니다.
        except OSError:
            need_copy = True  # 크기 확인이 실패하면 다시 복사합니다.

    if need_copy:  # 필요할 때만 복사합니다.
        shutil.copy2(src_path, dst_path)  # 원본을 임시 ASCII 경로로 복사합니다.

    return dst_path  # 사용 가능한 캐시 경로를 반환합니다.


def load_darknet_net(cfg_path: str, weights_path: str):
    try:
        return cv.dnn.readNetFromDarknet(cfg_path, weights_path)  # 먼저 원본 경로로 로드합니다.
    except cv.error as first_error:
        cfg_ascii = _ensure_ascii_copy(cfg_path)  # cfg 파일을 ASCII 경로로 복사합니다.
        weights_ascii = _ensure_ascii_copy(weights_path)  # weights 파일도 ASCII 경로로 복사합니다.
        try:
            print("[INFO] OpenCV DNN 경로 이슈 대응: ASCII 임시 경로에서 모델 로드를 재시도합니다.")  # 경로 우회 사실을 알립니다.
            return cv.dnn.readNetFromDarknet(cfg_ascii, weights_ascii)  # 복사본으로 다시 로드합니다.
        except cv.error:
            raise RuntimeError(
                "YOLO 모델 로드에 실패했습니다. cfg/weights 파일 경로 및 파일 손상을 확인하세요.\n"
                f"cfg={cfg_path}\nweights={weights_path}"
            ) from first_error  # 원래 오류를 유지한 채 예외를 올립니다.


def get_output_layer_names(net):
    if hasattr(net, "getUnconnectedOutLayersNames"):  # 최신 OpenCV API를 우선 사용합니다.
        return net.getUnconnectedOutLayersNames()  # 출력 레이어 이름 목록을 바로 반환합니다.
    layer_names = net.getLayerNames()  # 구버전 OpenCV용 레이어 이름을 가져옵니다.
    unconnected = net.getUnconnectedOutLayers()  # 연결되지 않은 출력 레이어 인덱스를 가져옵니다.
    return [layer_names[i - 1] for i in unconnected.flatten()]  # 인덱스를 이름으로 변환합니다.


def iou_xyxy(box_a, box_b):
    x1 = max(box_a[0], box_b[0])  # 교집합 왼쪽 경계입니다.
    y1 = max(box_a[1], box_b[1])  # 교집합 위쪽 경계입니다.
    x2 = min(box_a[2], box_b[2])  # 교집합 오른쪽 경계입니다.
    y2 = min(box_a[3], box_b[3])  # 교집합 아래쪽 경계입니다.

    inter_w = max(0.0, x2 - x1)  # 교집합 너비를 계산합니다.
    inter_h = max(0.0, y2 - y1)  # 교집합 높이를 계산합니다.
    inter = inter_w * inter_h  # 교집합 면적입니다.

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])  # 첫 번째 박스 면적입니다.
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])  # 두 번째 박스 면적입니다.
    union = area_a + area_b - inter  # 합집합 면적입니다.

    return inter / union if union > 1e-6 else 0.0  # IoU를 반환합니다.


def detect_objects_yolo(net, frame, conf_threshold=0.5, nms_threshold=0.4, input_size=416):
    h, w = frame.shape[:2]  # 프레임 높이와 너비를 얻습니다.

    blob = cv.dnn.blobFromImage(
        frame,
        scalefactor=1.0 / 255.0,
        size=(input_size, input_size),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)  # YOLO 입력 blob을 네트워크에 넣습니다.
    outputs = net.forward(get_output_layer_names(net))  # 출력 레이어의 결과를 추론합니다.

    boxes_xywh = []  # NMS 전 박스 목록을 저장합니다.
    confidences = []  # 각 박스의 신뢰도를 저장합니다.
    class_ids = []  # 각 박스의 클래스 ID를 저장합니다.

    for output in outputs:  # 각 출력 레이어를 순회합니다.
        for det in output:  # 각 검출 결과를 순회합니다.
            scores = det[5:]  # 클래스별 점수를 분리합니다.
            class_id = int(np.argmax(scores))  # 가장 높은 클래스 인덱스를 선택합니다.
            confidence = float(scores[class_id])  # 선택된 클래스의 점수를 신뢰도로 사용합니다.

            if confidence < conf_threshold:  # 임계값보다 낮으면 버립니다.
                continue

            cx = int(det[0] * w)  # 중심 x 좌표를 픽셀로 변환합니다.
            cy = int(det[1] * h)  # 중심 y 좌표를 픽셀로 변환합니다.
            bw = int(det[2] * w)  # 박스 너비를 픽셀로 변환합니다.
            bh = int(det[3] * h)  # 박스 높이를 픽셀로 변환합니다.

            x = int(cx - bw / 2)  # 좌상단 x 좌표를 계산합니다.
            y = int(cy - bh / 2)  # 좌상단 y 좌표를 계산합니다.

            boxes_xywh.append([x, y, bw, bh])  # NMS용 박스를 추가합니다.
            confidences.append(confidence)  # 신뢰도를 추가합니다.
            class_ids.append(class_id)  # 클래스 ID를 추가합니다.

    indices = cv.dnn.NMSBoxes(boxes_xywh, confidences, conf_threshold, nms_threshold)  # 중복 박스를 제거합니다.

    detections_xyxy = []  # 최종 검출 박스를 저장합니다.
    kept_class_ids = []  # 유지된 클래스 ID를 저장합니다.
    kept_confidences = []  # 유지된 신뢰도를 저장합니다.

    if len(indices) > 0:  # 남은 박스가 있으면 좌표를 변환합니다.
        for idx in indices.flatten():  # NMS가 선택한 인덱스를 순회합니다.
            x, y, bw, bh = boxes_xywh[idx]  # 좌표를 꺼냅니다.
            x1 = max(0, x)  # 프레임 범위를 벗어나지 않게 보정합니다.
            y1 = max(0, y)  # 프레임 범위를 벗어나지 않게 보정합니다.
            x2 = min(w - 1, x + bw)  # 오른쪽 경계를 보정합니다.
            y2 = min(h - 1, y + bh)  # 아래쪽 경계를 보정합니다.
            if x2 <= x1 or y2 <= y1:  # 잘못된 박스는 버립니다.
                continue
            detections_xyxy.append([float(x1), float(y1), float(x2), float(y2)])  # xyxy 형식으로 저장합니다.
            kept_class_ids.append(class_ids[idx])  # 클래스 ID를 함께 저장합니다.
            kept_confidences.append(confidences[idx])  # 신뢰도도 함께 저장합니다.

    if detections_xyxy:  # 검출 결과가 있으면 배열로 변환합니다.
        det_array = np.array(detections_xyxy, dtype=np.float32)
    else:
        det_array = np.empty((0, 4), dtype=np.float32)  # 없으면 빈 배열을 사용합니다.

    return det_array, kept_class_ids, kept_confidences  # 검출 결과를 반환합니다.


def convert_bbox_to_z(bbox):
    x1, y1, x2, y2 = bbox  # xyxy 박스를 분해합니다.
    w = x2 - x1  # 너비를 계산합니다.
    h = y2 - y1  # 높이를 계산합니다.
    cx = x1 + w / 2.0  # 중심 x를 계산합니다.
    cy = y1 + h / 2.0  # 중심 y를 계산합니다.
    s = w * h  # 면적을 계산합니다.
    r = w / h if h > 1e-6 else 0.0  # 종횡비를 계산합니다.
    return np.array([[cx], [cy], [s], [r]], dtype=np.float32)  # 칼만 필터 측정 벡터로 반환합니다.


def convert_x_to_bbox(x):
    cx, cy, s, r = x[0], x[1], x[2], x[3]  # 상태 벡터에서 중심과 크기 정보를 꺼냅니다.
    if s <= 0 or r <= 0:  # 유효하지 않은 상태는 빈 박스로 처리합니다.
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    w = np.sqrt(s * r)  # 너비를 복원합니다.
    h = s / w if w > 1e-6 else 0.0  # 높이를 복원합니다.
    x1 = cx - w / 2.0  # 좌상단 x를 계산합니다.
    y1 = cy - h / 2.0  # 좌상단 y를 계산합니다.
    x2 = cx + w / 2.0  # 우하단 x를 계산합니다.
    y2 = cy + h / 2.0  # 우하단 y를 계산합니다.
    return np.array([x1, y1, x2, y2], dtype=np.float32)  # xyxy 형식의 박스를 반환합니다.


class KalmanBoxTracker:
    count = 0  # 생성된 트랙터 ID를 전역적으로 세기 위한 카운터입니다.

    def __init__(self, bbox):
        self.kf = cv.KalmanFilter(7, 4)  # 7차 상태, 4차 관측의 칼만 필터를 만듭니다.

        self.kf.transitionMatrix = np.array(  # 상태 전이 행렬을 설정합니다.
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.kf.measurementMatrix = np.array(  # 관측 행렬을 설정합니다.
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2  # 예측 잡음을 설정합니다.
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1  # 관측 잡음을 설정합니다.
        self.kf.errorCovPost = np.eye(7, dtype=np.float32)  # 초기 오차 공분산을 설정합니다.

        z = convert_bbox_to_z(bbox)  # 초기 박스를 상태 벡터로 변환합니다.
        self.kf.statePost = np.array(  # 초기 상태를 설정합니다.
            [[z[0, 0]], [z[1, 0]], [z[2, 0]], [z[3, 0]], [0], [0], [0]],
            dtype=np.float32,
        )

        self.time_since_update = 0  # 마지막 갱신 이후 경과 프레임 수입니다.
        self.id = KalmanBoxTracker.count  # 트랙 고유 ID를 부여합니다.
        KalmanBoxTracker.count += 1  # 다음 ID를 위해 카운터를 증가시킵니다.
        self.history = []  # 예측 이력을 저장합니다.
        self.hits = 0  # 누적 업데이트 횟수입니다.
        self.hit_streak = 0  # 연속 매칭 횟수입니다.
        self.age = 0  # 트랙의 총 경과 프레임 수입니다.

    def update(self, bbox):
        self.time_since_update = 0  # 갱신되었으므로 경과 프레임을 초기화합니다.
        self.history.clear()  # 이전 예측 이력을 지웁니다.
        self.hits += 1  # 누적 히트를 증가시킵니다.
        self.hit_streak += 1  # 연속 히트도 증가시킵니다.
        z = convert_bbox_to_z(bbox)  # 관측 벡터로 변환합니다.
        self.kf.correct(z)  # 칼만 필터를 보정합니다.

    def predict(self):
        if (self.kf.statePost[6, 0] + self.kf.statePost[2, 0]) <= 0:  # 비정상적인 스케일 감소를 방지합니다.
            self.kf.statePost[6, 0] = 0  # 속도 성분을 0으로 둡니다.

        prediction = self.kf.predict()  # 다음 상태를 예측합니다.
        self.age += 1  # 트랙의 나이를 증가시킵니다.

        if self.time_since_update > 0:  # 매칭이 끊기면 연속 히트를 초기화합니다.
            self.hit_streak = 0

        self.time_since_update += 1  # 업데이트가 없었음을 기록합니다.
        pred_bbox = convert_x_to_bbox(prediction[:, 0])  # 예측 상태를 박스로 복원합니다.
        self.history.append(pred_bbox)  # 예측 이력에 저장합니다.
        return pred_bbox  # 예측 박스를 반환합니다.

    def get_state(self):
        return convert_x_to_bbox(self.kf.statePost[:, 0])  # 현재 상태를 박스로 반환합니다.


def greedy_assignment(iou_matrix):
    matches = []  # 선택된 매칭 쌍을 저장합니다.
    if iou_matrix.size == 0:  # 비교할 행렬이 없으면 빈 결과를 반환합니다.
        return np.empty((0, 2), dtype=int)

    used_rows = set()  # 이미 사용한 검출 인덱스를 기록합니다.
    used_cols = set()  # 이미 사용한 트랙 인덱스를 기록합니다.

    flat_indices = np.argsort(-iou_matrix, axis=None)  # IoU가 큰 순서로 정렬합니다.
    rows, cols = np.unravel_index(flat_indices, iou_matrix.shape)  # 평면 인덱스를 행/열로 바꿉니다.

    for r, c in zip(rows, cols):  # 높은 IoU부터 순회합니다.
        if r in used_rows or c in used_cols:  # 이미 사용한 행/열이면 건너뜁니다.
            continue
        used_rows.add(r)  # 행을 사용 처리합니다.
        used_cols.add(c)  # 열을 사용 처리합니다.
        matches.append([r, c])  # 매칭 결과를 추가합니다.

    if not matches:  # 매칭이 없으면 빈 배열을 반환합니다.
        return np.empty((0, 2), dtype=int)
    return np.array(matches, dtype=int)  # 매칭 결과를 배열로 반환합니다.


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:  # 기존 트랙이 없으면 검출은 모두 새 트랙 후보입니다.
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),  # 모든 검출이 미매칭입니다.
            np.empty((0,), dtype=int),  # 미매칭 트랙은 없습니다.
        )

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)  # IoU 행렬을 만듭니다.
    for d, det in enumerate(detections):  # 각 검출 박스에 대해
        for t, trk in enumerate(trackers):  # 각 트랙 박스와 비교합니다.
            iou_matrix[d, t] = iou_xyxy(det, trk)  # IoU 값을 채웁니다.

    if SCIPY_AVAILABLE:  # scipy가 있으면 최적 매칭을 수행합니다.
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # IoU를 최대화하는 매칭을 구합니다.
        matched_indices = np.stack([row_ind, col_ind], axis=1) if len(row_ind) > 0 else np.empty((0, 2), dtype=int)
    else:
        matched_indices = greedy_assignment(iou_matrix)  # 없으면 greedy 방식으로 매칭합니다.

    unmatched_detections = []  # 매칭되지 않은 검출을 저장합니다.
    for d in range(len(detections)):  # 모든 검출을 확인합니다.
        if d not in matched_indices[:, 0] if len(matched_indices) > 0 else True:  # 매칭되지 않은 검출이면
            unmatched_detections.append(d)  # 미매칭 목록에 추가합니다.

    unmatched_trackers = []  # 매칭되지 않은 트랙을 저장합니다.
    for t in range(len(trackers)):  # 모든 트랙을 확인합니다.
        if t not in matched_indices[:, 1] if len(matched_indices) > 0 else True:  # 매칭되지 않은 트랙이면
            unmatched_trackers.append(t)  # 미매칭 목록에 추가합니다.

    matches = []  # IoU 기준을 통과한 최종 매칭을 저장합니다.
    for m in matched_indices:  # 후보 매칭을 검사합니다.
        if iou_matrix[m[0], m[1]] < iou_threshold:  # 임계값보다 낮으면
            unmatched_detections.append(m[0])  # 검출을 미매칭으로 돌립니다.
            unmatched_trackers.append(m[1])  # 트랙도 미매칭으로 돌립니다.
        else:
            matches.append(m)  # 임계값을 통과한 매칭만 유지합니다.

    if len(matches) == 0:  # 매칭이 없으면 빈 배열로 정리합니다.
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches, dtype=int)  # 리스트를 배열로 바꿉니다.

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)  # 결과를 반환합니다.


class SortTracker:
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age  # 트랙 삭제까지 허용할 최대 미갱신 프레임 수입니다.
        self.min_hits = min_hits  # 확정 트랙이 되기 위한 최소 히트 수입니다.
        self.iou_threshold = iou_threshold  # 검출-트랙 매칭 기준 IoU입니다.
        self.trackers = []  # 현재 활성 트랙 목록입니다.
        self.frame_count = 0  # 누적 프레임 카운터입니다.

    def update(self, detections):
        self.frame_count += 1  # 프레임 카운터를 증가시킵니다.

        trks = np.zeros((len(self.trackers), 4), dtype=np.float32)  # 예측된 트랙 박스를 담을 배열입니다.
        to_del = []  # 제거할 트랙 인덱스를 기록합니다.

        for t, trk in enumerate(self.trackers):  # 모든 트랙에 대해 예측을 수행합니다.
            pos = trk.predict()  # 다음 프레임의 위치를 예측합니다.
            trks[t] = pos  # 예측값을 배열에 저장합니다.
            if np.any(np.isnan(pos)):  # NaN이 있으면 트랙 제거 후보로 둡니다.
                to_del.append(t)

        for t in reversed(to_del):  # 뒤에서부터 지워 인덱스 꼬임을 방지합니다.
            self.trackers.pop(t)  # 비정상 트랙을 제거합니다.

        trks = np.array([trk.get_state() for trk in self.trackers], dtype=np.float32) if self.trackers else np.empty((0, 4), dtype=np.float32)  # 현재 트랙 상태를 다시 모읍니다.

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections,
            trks,
            self.iou_threshold,
        )

        for m in matched:  # 매칭된 쌍은 기존 트랙을 갱신합니다.
            self.trackers[m[1]].update(detections[m[0]])

        for i in unmatched_dets:  # 새로 등장한 객체는 새 트랙으로 생성합니다.
            self.trackers.append(KalmanBoxTracker(detections[i]))

        results = []  # 화면에 내보낼 확정 트랙을 저장합니다.
        for trk in reversed(self.trackers):  # 제거 가능성을 고려해 뒤에서부터 순회합니다.
            d = trk.get_state()  # 현재 트랙 박스를 가져옵니다.
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  # 확정 조건을 검사합니다.
                results.append(np.concatenate([d, [trk.id + 1]], axis=0))  # 박스 뒤에 ID를 붙입니다.
            if trk.time_since_update > self.max_age:  # 오래 갱신되지 않은 트랙은 제거합니다.
                self.trackers.remove(trk)

        if len(results) > 0:  # 결과가 있으면 2D 배열로 반환합니다.
            return np.stack(results, axis=0)
        return np.empty((0, 5), dtype=np.float32)  # 없으면 빈 배열을 반환합니다.


def get_track_color(track_id):
    rng = np.random.default_rng(track_id * 9973)  # ID별로 재현 가능한 색을 만듭니다.
    color = rng.integers(80, 255, size=3)  # 너무 어둡지 않은 색을 고릅니다.
    return int(color[0]), int(color[1]), int(color[2])  # BGR 색상으로 반환합니다.


def assign_class_to_tracks(tracks, detections, class_ids, confidences):
    track_info = {}  # 트랙별 클래스 정보를 저장합니다.
    if len(tracks) == 0 or len(detections) == 0:  # 비교 대상이 없으면 바로 반환합니다.
        return track_info

    for trk in tracks:  # 각 추적 박스에 대해
        tx1, ty1, tx2, ty2, tid = trk  # 박스와 트랙 ID를 분해합니다.
        best_iou = 0.0  # 가장 좋은 IoU를 저장합니다.
        best_idx = -1  # 가장 잘 맞는 검출 인덱스를 저장합니다.
        for i, det in enumerate(detections):  # 모든 검출과 비교합니다.
            score = iou_xyxy([tx1, ty1, tx2, ty2], det)  # IoU를 계산합니다.
            if score > best_iou:  # 더 좋은 매칭이면 갱신합니다.
                best_iou = score
                best_idx = i

        if best_idx >= 0 and best_iou > 0.1:  # 충분히 겹치는 경우만 클래스 정보를 부여합니다.
            track_info[int(tid)] = {
                "class_id": int(class_ids[best_idx]),  # 클래스 ID를 저장합니다.
                "confidence": float(confidences[best_idx]),  # 신뢰도도 저장합니다.
            }

    return track_info  # 트랙별 부가 정보를 반환합니다.


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치를 기준 디렉터리로 사용합니다.

    parser = argparse.ArgumentParser(description="YOLOv3 + SORT multi-object tracking")  # 인자 파서를 만듭니다.
    parser.add_argument("--video", type=str, default=os.path.join(script_dir, "slow_traffic_small.mp4"), help="Input video path")  # 입력 비디오 경로입니다.
    parser.add_argument("--cfg", type=str, default=os.path.join(script_dir, "yolov3.cfg"), help="YOLOv3 config path")  # cfg 파일 경로입니다.
    parser.add_argument("--weights", type=str, default=os.path.join(script_dir, "yolov3.weights"), help="YOLOv3 weights path")  # weights 파일 경로입니다.
    parser.add_argument("--names", type=str, default=os.path.join(script_dir, "coco.names"), help="Class names path")  # 클래스 이름 파일 경로입니다.
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Detection confidence threshold")  # 검출 신뢰도 임계값입니다.
    parser.add_argument("--nms-thres", type=float, default=0.4, help="NMS IoU threshold")  # NMS 임계값입니다.
    parser.add_argument("--iou-thres", type=float, default=0.3, help="SORT association IoU threshold")  # SORT 매칭 IoU 임계값입니다.
    parser.add_argument("--max-age", type=int, default=15, help="Max missing frames before track deletion")  # 트랙 삭제 허용 프레임 수입니다.
    parser.add_argument("--min-hits", type=int, default=3, help="Min hits to confirm a track")  # 트랙 확정 최소 히트 수입니다.
    parser.add_argument("--input-size", type=int, default=416, help="YOLO input size")  # YOLO 입력 크기입니다.
    parser.add_argument("--save", type=str, default="", help="Optional output video path")  # 저장 경로를 지정할 수 있습니다.
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA backend if available")  # CUDA 사용 옵션입니다.
    return parser.parse_args()  # 인자를 파싱해 반환합니다.


def main():
    args = parse_args()  # 커맨드라인 인자를 읽습니다.

    if not os.path.exists(args.video):  # 입력 비디오가 없으면 중단합니다.
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {args.video}")
    if not os.path.exists(args.cfg):  # cfg 파일이 없으면 중단합니다.
        raise FileNotFoundError(f"YOLO cfg 파일을 찾을 수 없습니다: {args.cfg}")
    if not os.path.exists(args.weights):  # weights 파일이 없으면 중단합니다.
        raise FileNotFoundError(f"YOLO weights 파일을 찾을 수 없습니다: {args.weights}")

    class_names = load_class_names(args.names)  # 클래스 이름 목록을 불러옵니다.

    net = load_darknet_net(args.cfg, args.weights)  # YOLO 네트워크를 로드합니다.
    if args.use_cuda:  # CUDA 옵션이 켜져 있으면 GPU 백엔드를 사용합니다.
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)  # 기본 OpenCV 백엔드를 사용합니다.
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)  # CPU 추론으로 설정합니다.

    cap = cv.VideoCapture(args.video)  # 비디오 파일을 엽니다.
    if not cap.isOpened():  # 열기 실패 시 중단합니다.
        raise RuntimeError(f"비디오를 열 수 없습니다: {args.video}")

    writer = None  # 저장하지 않을 때는 writer가 없습니다.
    if args.save:  # 저장 경로가 지정되면 파일로 기록합니다.
        fourcc = cv.VideoWriter_fourcc(*"mp4v")  # mp4 인코더를 사용합니다.
        fps_out = cap.get(cv.CAP_PROP_FPS)  # 원본 FPS를 읽습니다.
        if fps_out <= 0:  # FPS를 읽지 못하면 기본값을 사용합니다.
            fps_out = 25.0
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))  # 비디오 너비를 읽습니다.
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  # 비디오 높이를 읽습니다.
        writer = cv.VideoWriter(args.save, fourcc, fps_out, (width, height))  # 출력 비디오 writer를 만듭니다.

    tracker = SortTracker(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_thres)  # SORT 추적기를 초기화합니다.

    prev_time = time.perf_counter()  # FPS 계산용 시작 시간을 저장합니다.
    smooth_fps = 0.0  # 부드러운 FPS 값을 저장합니다.

    print("실행 중: 종료하려면 q 또는 ESC를 누르세요.")  # 사용자 안내 메시지를 출력합니다.

    while True:  # 비디오가 끝나거나 종료 키가 눌릴 때까지 반복합니다.
        ret, frame = cap.read()  # 한 프레임을 읽습니다.
        if not ret:  # 더 이상 읽을 프레임이 없으면 종료합니다.
            break

        detections, class_ids, confidences = detect_objects_yolo(
            net,
            frame,
            conf_threshold=args.conf_thres,
            nms_threshold=args.nms_thres,
            input_size=args.input_size,
        )

        tracks = tracker.update(detections)  # 검출 결과를 추적기에 반영합니다.
        track_extra = assign_class_to_tracks(tracks, detections, class_ids, confidences)  # 트랙에 클래스를 연결합니다.

        for trk in tracks:  # 각 추적 결과를 화면에 그립니다.
            x1, y1, x2, y2, tid = trk.astype(int)  # 좌표와 ID를 정수로 변환합니다.
            color = get_track_color(tid)  # ID 기반 색상을 선택합니다.

            label = f"ID {tid}"  # 기본 라벨은 ID만 표시합니다.
            if tid in track_extra:  # 클래스 정보가 있으면 함께 표시합니다.
                cid = track_extra[tid]["class_id"]  # 클래스 ID를 꺼냅니다.
                conf = track_extra[tid]["confidence"]  # 신뢰도를 꺼냅니다.
                cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)  # 클래스 이름을 찾습니다.
                label = f"ID {tid} {cname} {conf:.2f}"  # 라벨 문자열을 구성합니다.

            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 객체 경계 상자를 그립니다.
            cv.putText(frame, label, (x1, max(20, y1 - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)  # 라벨을 표시합니다.

        now = time.perf_counter()  # 현재 시간을 측정합니다.
        inst_fps = 1.0 / max(now - prev_time, 1e-6)  # 순간 FPS를 계산합니다.
        prev_time = now  # 다음 계산을 위해 시간을 갱신합니다.
        smooth_fps = inst_fps if smooth_fps == 0.0 else (0.9 * smooth_fps + 0.1 * inst_fps)  # FPS를 평활화합니다.

        info_text = f"FPS: {smooth_fps:.1f} | Dets: {len(detections)} | Tracks: {len(tracks)}"  # 상태 정보를 만듭니다.
        cv.putText(frame, info_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # 상태 정보를 표시합니다.

        if writer is not None:  # 저장 옵션이 있으면 파일에 기록합니다.
            writer.write(frame)

        cv.imshow("YOLOv3 + SORT Multi-Object Tracking", frame)  # 추적 결과를 화면에 출력합니다.
        key = cv.waitKey(1) & 0xFF  # 키 입력을 확인합니다.
        if key in (27, ord("q")):  # ESC 또는 q를 누르면 종료합니다.
            break

    cap.release()  # 비디오 캡처를 해제합니다.
    if writer is not None:  # 저장 중이었다면 writer도 해제합니다.
        writer.release()
    cv.destroyAllWindows()  # 모든 창을 닫습니다.


if __name__ == "__main__":
    main()
