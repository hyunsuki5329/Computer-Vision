import cv2 as cv
import sys

# 1. 초기 설정
img = cv.imread('soccer.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

# 이미지 크기 0.5배로 축소
img = cv.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)

brush_size = 5  # 초기 붓 크기
L_color, R_color = (255, 0, 0), (0, 0, 255)  # 파란색(좌클릭), 빨간색(우클릭)

# 2. 마우스 콜백 함수 정의
def draw(event, x, y, flags, param):
    global brush_size
    
    # 좌클릭 또는 드래그 중 좌클릭 상태일 때 파란색 원 그리기
    if event == cv.EVENT_LBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON)):
        cv.circle(img, (x, y), brush_size, L_color, -1)
    
    # 우클릭 또는 드래그 중 우클릭 상태일 때 빨간색 원 그리기
    elif event == cv.EVENT_RBUTTONDOWN or (event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_RBUTTON)):
        cv.circle(img, (x, y), brush_size, R_color, -1)
    
    cv.imshow('Painting', img)

# 3. 윈도우 생성 및 콜백 등록
cv.namedWindow('Painting')
cv.imshow('Painting', img)
cv.setMouseCallback('Painting', draw)

# 4. 키보드 입력 루프
while True:
    key = cv.waitKey(1) & 0xFF  # 1ms 대기하며 키 입력 받기
    
    if key == ord('q'):  # 'q' 누르면 종료
        break
    elif key == ord('+'):  # '+' 누르면 크기 증가 (최대 15)
        brush_size = min(brush_size + 1, 15)
        print(f"현재 붓 크기: {brush_size}")
    elif key == ord('-'):  # '-' 누르면 크기 감소 (최소 1)
        brush_size = max(brush_size - 1, 1)
        print(f"현재 붓 크기: {brush_size}")

cv.destroyAllWindows()