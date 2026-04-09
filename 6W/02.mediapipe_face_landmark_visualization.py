import cv2 as cv  # 웹캠 캡처와 랜드마크 시각화를 위해 OpenCV를 사용합니다.

from google.protobuf import message_factory  # protobuf 메시지 팩토리를 사용합니다.
from google.protobuf import symbol_database  # protobuf 심볼 데이터베이스를 사용합니다.


if not hasattr(symbol_database.SymbolDatabase, "GetPrototype"):  # protobuf 6 호환 패치를 적용합니다.
    def _symbol_database_get_prototype(self, descriptor):
        return message_factory.GetMessageClass(descriptor)  # descriptor로부터 메시지 클래스를 얻습니다.


    symbol_database.SymbolDatabase.GetPrototype = _symbol_database_get_prototype  # 누락된 메서드를 주입합니다.


if not hasattr(message_factory.MessageFactory, "GetPrototype"):  # 구버전 API가 없으면 대체합니다.
    def _message_factory_get_prototype(self, descriptor):
        return message_factory.GetMessageClass(descriptor)  # 메시지 클래스를 직접 반환합니다.


    message_factory.MessageFactory.GetPrototype = _message_factory_get_prototype  # 호환성 메서드를 추가합니다.

try:
    import mediapipe as mp  # FaceMesh를 사용하기 위해 MediaPipe를 불러옵니다.
except ImportError as exc:
    raise ImportError(
        "mediapipe is not installed. Install with: pip install mediapipe"
    ) from exc  # 설치가 안 되어 있으면 명확한 메시지로 중단합니다.


def draw_face_landmarks(frame_bgr, face_landmarks):
    h, w = frame_bgr.shape[:2]  # 프레임의 높이와 너비를 구합니다.
    for lm in face_landmarks.landmark:  # 얼굴의 모든 랜드마크를 순회합니다.
        x = int(lm.x * w)  # 정규화된 x 좌표를 픽셀 좌표로 변환합니다.
        y = int(lm.y * h)  # 정규화된 y 좌표를 픽셀 좌표로 변환합니다.
        if 0 <= x < w and 0 <= y < h:  # 프레임 범위 안의 점만 그립니다.
            cv.circle(frame_bgr, (x, y), 1, (0, 255, 0), -1)  # 랜드마크를 초록 점으로 표시합니다.


def main():
    mp_face_mesh = mp.solutions.face_mesh  # FaceMesh 솔루션 모듈을 준비합니다.

    cap = cv.VideoCapture(0)  # 기본 웹캠(0번 장치)을 엽니다.
    if not cap.isOpened():  # 웹캠 열기에 실패하면 오류를 발생시킵니다.
        raise RuntimeError("Could not open webcam (index 0).")

    with mp_face_mesh.FaceMesh(  # 얼굴 랜드마크 검출기를 초기화합니다.
        static_image_mode=False,  # 비디오 스트림 모드로 처리합니다.
        max_num_faces=1,  # 한 번에 최대 1개의 얼굴만 추적합니다.
        refine_landmarks=False,  # 세부 눈/입 보정은 끕니다.
        min_detection_confidence=0.5,  # 얼굴 탐지 최소 신뢰도입니다.
        min_tracking_confidence=0.5,  # 추적 최소 신뢰도입니다.
    ) as face_mesh:
        while True:  # 종료 키가 눌릴 때까지 반복합니다.
            ret, frame_bgr = cap.read()  # 웹캠에서 한 프레임을 읽습니다.
            if not ret:  # 프레임을 읽지 못하면 종료합니다.
                break

            frame_bgr = cv.flip(frame_bgr, 1)  # 거울처럼 보이도록 좌우 반전합니다.
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)  # MediaPipe 입력용 RGB로 변환합니다.
            results = face_mesh.process(frame_rgb)  # 얼굴 랜드마크를 검출합니다.

            if results.multi_face_landmarks:  # 얼굴이 검출되었으면
                for face_landmarks in results.multi_face_landmarks:  # 검출된 모든 얼굴을 순회합니다.
                    draw_face_landmarks(frame_bgr, face_landmarks)  # 랜드마크를 프레임 위에 그립니다.

            cv.imshow("MediaPipe FaceMesh (ESC to quit)", frame_bgr)  # 결과 화면을 표시합니다.

            key = cv.waitKey(1) & 0xFF  # 키 입력을 확인합니다.
            if key == 27:  # ESC 키가 눌리면 종료합니다.
                break

    cap.release()  # 웹캠 자원을 해제합니다.
    cv.destroyAllWindows()  # 열린 모든 창을 닫습니다.


if __name__ == "__main__":
    main()
