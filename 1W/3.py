import cv2 as cv
import sys

# 1. 이미지 로드 및 초기화
img = cv.imread('soccer.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 이미지 크기 조절 (화면에 맞게 0.5배 축소)
img = cv.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
ori_img = img.copy()  # 리셋('r')을 위한 원본 복사본

# 전역 변수
ix, iy = -1, -1  # 마우스 클릭 시작 좌표
drawing = False
roi = None  # 선택된 영역을 저장할 변수

# 2. 마우스 콜백 함수 정의
def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, img, roi

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            # 드래그 중 실시간 사각형 표시
            img_draw = img.copy()
            cv.rectangle(img_draw, (ix, iy), (x, y), (0, 255, 0), 2) 
            cv.imshow('Select ROI', img_draw)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        # 사각형 그리기 확정
        cv.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv.imshow('Select ROI', img)
        
        # ROI 추출
        # 좌표의 선후 관계가 바뀔 수 있으므로 min, max 사용 
        x1, x2 = min(ix, x), max(ix, x)
        y1, y2 = min(iy, y), max(iy, y)
        
        if x1 != x2 and y1 != y2:
            roi = ori_img[y1:y2, x1:x2]  # 원본 복사본에서 영역 추출
            cv.imshow('Cropped ROI', roi) # 별도의 창에 출력

# 3. 윈도우 생성 및 콜백 등록
cv.namedWindow('Select ROI')
cv.imshow('Select ROI', img)
cv.setMouseCallback('Select ROI', draw_roi)

# 4. 키보드 이벤트 처리 루프
while True:
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('q'): # 종료
        break
        
    elif key == ord('r'): # 영역 선택 리셋
        img = ori_img.copy()
        roi = None
        cv.imshow('Select ROI', img)
        if cv.getWindowProperty('Cropped ROI', 0) >= 0:
            cv.destroyWindow('Cropped ROI')
        print("영역 선택이 리셋되었습니다.")
        
    elif key == ord('s'): # 선택 영역 저장
        if roi is not None:
            cv.imwrite('soccer_roi.jpg', roi)
            print("선택 영역이 'soccer_roi.jpg'로 저장되었습니다.")
        else:
            print("저장할 영역이 선택되지 않았습니다.")

cv.destroyAllWindows()