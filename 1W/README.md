1. 이미지 불러오기 및 그레이스케일 변환

설명
• OpenCV를 사용하여 이미지를 불러오고 화면에 출력
• 원본 이미지와 그레이스케일로 변환된 이미지를 나란히 표시

요구사항
• cv.imread()를 사용하여 이미지 로드
• cv.cvtColor() 함수를 사용해 이미지를 그레이스케일로 변환
• np.hstack() 함수를 이용해 원본 이미지와 그레이스케일 이미지를 가로로 연결하여 출력
• cv.imshow()와 cv.waitKey()를 사용해 결과를 화면에 표시하고, 아무 키나 누르면 창이 닫히도록 할 것

힌트
• OpenCV는 이미지를 BGR 형식으로 읽음
• 그레이스케일 변환시 cv.COLOR_BGR2GRAY 사용

코드
```python
import cv2 as cv
import sys
import numpy as np

# 1. 이미지 불러오기
img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

# 2. 이미지 크기 줄이기 (0.5배 축소)
# fx, fy 인자를 사용하여 가로세로 비율을 조절합니다.
img_small = cv.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)

# 3. 그레이스케일 변환 (축소된 이미지 기준)
gray = cv.cvtColor(img_small, cv.COLOR_BGR2GRAY)

# 4. hstack을 위해 그레이스케일 이미지를 3채널로 변환
gray_3channel = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

# 5. 원본(축소본)과 그레이스케일을 가로로 연결
combined = np.hstack((img_small, gray_3channel))

# 6. 결과 화면 출력
cv.imshow('Reduced Size - Color & Gray', combined)

# 속성 출력으로 크기 확인
print(f"Original Size: {img.shape}")
print(f"Reduced Size: {img_small.shape}")

cv.waitKey()
cv.destroyAllWindows()
```

출력
![alt text](image1.png)



2. 페인팅 붓 크기 조절 기능 추가

설명
• 마우스 입력으로 이미지 위에 붓질
• 키보드 입력을 이용해 붓의 크기를 조절하는 기능 추가

요구사항
• 초기 붓 크기는 5를 사용
• + 입력 시 붓 크기 1 증가, - 입력 시 붓 크기 1 감소
• 붓 크기는 최소 1, 최대 15로 제한
• 좌클릭=파란색, 우클릭=빨간색, 드래그로 연속 그리기
• q 키를 누르면 영상 창이 종료

힌트
• 마우스 이벤트는 cv.setMouseCallback()을 통해 처리하며, cv.circle()을 이용해 현재 붓 크기로 원을 그림
• cv.waitKey(1)로 받은 값을 이용해 +, -, q를 구분
• Key 입력은 루프 안에서 처리

코드
```python
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
```

출력
![alt text](image2.png)



3. 마우스로 영역 선택 및 ROI(관심영역) 추출

설명
• 이미지를 불러오고 사용자가 마우스로 클릭하고 드래그하여 관심영역(ROI)을 선택
• 선택한 영역만 따로 저장하거나 표시

요구사항
• 이미지를 불러오고 화면에 출력
• cv.setMouseCallback()을 사용하여 마우스 이벤트를 처리
• 사용자가 클릭한 시작점에서 드래그하여 사각형을 그리며 영역을 선택
• 마우스를 놓으면 해당 영역을 잘라내서 별도의 창에 출력
• r 키를 누르면 영역 선택을 리셋하고 처음부터 다시 선택
• s 키를 누르면 선택한 영역을 이미지 파일로 저장

힌트
• cv.rectangle() 함수로 드래그 중인 영역을 시각화
• ROI 추출은 numpy 슬라이싱을 사용
• cv.imwrite()를 사용하여 이미지를 저장

코드
```python
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
```

출력
![alt text](image3.png)