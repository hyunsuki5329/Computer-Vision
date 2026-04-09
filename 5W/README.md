# 1. MNIST 기반 간단한 이미지 분류기 구현

- MNIST 손글씨 숫자 데이터셋을 로드
- 전체 데이터를 하나로 합친 뒤 8:2로 재분할
- MLP(완전연결 신경망) 모델을 구성해 학습
import numpy as np  # 배열 연산과 셔플 처리를 위해 NumPy를 사용합니다.
import tensorflow as tf  # MNIST 데이터 로드와 모델 학습을 위해 TensorFlow를 사용합니다.
from tensorflow.keras import layers, models  # Keras의 레이어와 모델 API를 가져옵니다.
	<summary>전체 코드</summary>

```python
	rng = np.random.default_rng(seed)  # 재현 가능한 셔플을 위한 난수 생성기를 만듭니다.
	indices = rng.permutation(len(images))  # 전체 샘플 인덱스를 무작위로 섞습니다.
	split_idx = int(len(images) * train_ratio)  # train/test를 나눌 기준 지점을 계산합니다.

	train_idx = indices[:split_idx]  # 앞부분 인덱스를 훈련용으로 사용합니다.
	test_idx = indices[split_idx:]  # 뒷부분 인덱스를 테스트용으로 사용합니다.
	rng = np.random.default_rng(seed)  # 재현 가능한 셔플을 위해 고정 시드 난수 생성기를 만듭니다.
	x_train = images[train_idx]  # 훈련 이미지 데이터를 추출합니다.
	y_train = labels[train_idx]  # 훈련 라벨 데이터를 추출합니다.
	x_test = images[test_idx]  # 테스트 이미지 데이터를 추출합니다.
	y_test = labels[test_idx]  # 테스트 라벨 데이터를 추출합니다.
	return x_train, x_test, y_train, y_test  # 분할된 데이터를 반환합니다.

	x_train = images[train_idx]  # 훈련 이미지 데이터를 인덱스로 추출합니다.
	y_train = labels[train_idx]  # 훈련 정답 라벨을 인덱스로 추출합니다.
	return images.astype("float32") / 255.0  # 픽셀값을 0~1로 정규화합니다.
	y_test = labels[test_idx]  # 테스트 정답 라벨을 인덱스로 추출합니다.
	return x_train, x_test, y_train, y_test  # 분할된 데이터 4개를 반환합니다.

	model = models.Sequential(  # 순차형 MLP 모델을 구성합니다.
def preprocess_images(images):
			layers.Flatten(input_shape=(28, 28)),  # 28x28 이미지를 1차원으로 펼칩니다.
			layers.Dense(128, activation="relu"),  # 첫 번째 은닉층입니다.
			layers.Dense(64, activation="relu"),  # 두 번째 은닉층입니다.
			layers.Dense(10, activation="softmax"),  # 10개 클래스 확률을 출력합니다.
	model = models.Sequential(  # 층을 순서대로 쌓는 Sequential 모델을 생성합니다.
		[
			layers.Flatten(input_shape=(28, 28)),  # 28x28 이미지를 1차원 벡터(784)로 펼칩니다.
	model.compile(  # 학습 설정을 컴파일합니다.
		optimizer="adam",  # Adam 최적화기를 사용합니다.
		loss="sparse_categorical_crossentropy",  # 정수 라벨용 다중분류 손실입니다.
		metrics=["accuracy"],  # 정확도를 함께 확인합니다.
	)
	return model  # 컴파일된 모델을 반환합니다.
	model.compile(  # 학습에 사용할 최적화/손실/평가 지표를 설정합니다.
		optimizer="adam",  # 적응형 학습률을 사용하는 Adam 옵티마이저를 사용합니다.
		loss="sparse_categorical_crossentropy",  # 정답이 정수 라벨일 때 쓰는 다중분류 손실입니다.
	(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()  # MNIST 데이터를 불러옵니다.
	)
	all_images = np.concatenate([x_train_raw, x_test_raw], axis=0)  # 전체 이미지를 하나로 합칩니다.
	all_labels = np.concatenate([y_train_raw, y_test_raw], axis=0)  # 전체 라벨도 하나로 합칩니다.

	x_train, x_test, y_train, y_test = split_dataset(all_images, all_labels, train_ratio=0.8, seed=42)  # 8:2로 재분할합니다.
	(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()  # MNIST 기본 train/test 데이터를 로드합니다.
	x_train = preprocess_images(x_train)  # 훈련 이미지를 정규화합니다.
	x_test = preprocess_images(x_test)  # 테스트 이미지도 정규화합니다.
	all_labels = np.concatenate([y_train_raw, y_test_raw], axis=0)  # 전체 라벨도 같은 순서로 하나로 합칩니다.
	print(f"Train set: {x_train.shape}, labels: {y_train.shape}")  # 훈련 데이터 크기를 출력합니다.
	print(f"Test set:  {x_test.shape}, labels: {y_test.shape}")  # 테스트 데이터 크기를 출력합니다.

	model = build_model()  # 모델 구조를 생성합니다.
	model.summary()  # 층별 파라미터 정보를 출력합니다.

	model.fit(  # 모델 학습을 시작합니다.
		x_train,  # 입력 이미지입니다.
		y_train,  # 정답 라벨입니다.
		epochs=5,  # 전체 데이터를 5번 반복 학습합니다.
		batch_size=128,  # 한 번에 처리할 샘플 수입니다.
		validation_split=0.1,  # 훈련 데이터의 일부를 검증용으로 분리합니다.
		verbose=1,  # 학습 로그를 자세히 출력합니다.
		x_train,  # 입력 훈련 이미지 데이터입니다.
		y_train,  # 입력 훈련 정답 라벨입니다.
	test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)  # 테스트셋으로 평가합니다.
	print(f"\nTest loss: {test_loss:.4f}")  # 테스트 손실을 출력합니다.
	print(f"Test accuracy: {test_accuracy:.4f}")  # 테스트 정확도를 출력합니다.
		verbose=1,  # 학습 진행 로그를 자세히 출력합니다.
	probabilities = model.predict(x_test, verbose=0)  # 각 샘플의 클래스 확률을 예측합니다.
	predictions = np.argmax(probabilities, axis=1)  # 가장 높은 확률의 클래스를 선택합니다.
	confusion = tf.math.confusion_matrix(y_test, predictions, num_classes=10).numpy()  # 혼동행렬을 계산합니다.
	print(f"\nTest loss: {test_loss:.4f}")  # 테스트 손실 값을 소수점 4자리로 출력합니다.
	print("\nConfusion matrix (rows=true label, cols=predicted label):")  # 혼동행렬의 의미를 안내합니다.
	print(confusion)  # 계산된 혼동행렬을 출력합니다.
	probabilities = model.predict(x_test, verbose=0)  # 각 테스트 샘플의 클래스별 확률을 예측합니다.
	predictions = np.argmax(probabilities, axis=1)  # 가장 확률이 높은 클래스를 최종 예측값으로 선택합니다.
	confusion = tf.math.confusion_matrix(y_test, predictions, num_classes=10).numpy()  # 10개 클래스 혼동행렬을 계산합니다.

	print("\nConfusion matrix (rows=true label, cols=predicted label):")  # 혼동행렬의 행/열 의미를 안내합니다.
	print(confusion)  # 계산된 혼동행렬 숫자를 출력합니다.


if __name__ == "__main__":
	main()
```

</details>

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

### 실행 결과

```text
Test loss: 0.0868
Test accuracy: 0.9731

Confusion matrix (rows=true label, cols=predicted label):
[[1376    0    1    2    0    3    6    0   12    1]
 [   0 1557    8    1    4    2    2    6    6    1]
 [   5    2 1348   20    2    1    2   12   13    0]
 [   1    0    5 1389    0   13    0    5   10    2]
 [   0    2    2    2 1306    1    4    3    2   15]
 [   4    0    1   16    0 1204   12    1    5    1]
 [   4    2    1    1    3    6 1358    0    3    0]
 [   2    0    7    4    4    0    0 1418    5    9]
 [   5    6    3   12    2   13    4    3 1299    4]
 [   4    2    0   12    9    4    1   15    7 1369]]
```


# 2. CIFAR-10 기반 CNN 이미지 분류 + dog.jpg 예측

- CIFAR-10 데이터셋 로드 및 정규화 전처리 수행
- CNN + BatchNorm + Dropout 구조로 분류 모델 구성
- 데이터 증강(RandomFlip/Rotation/Zoom/Contrast) 적용
- ReduceLROnPlateau와 EarlyStopping으로 학습 안정화
- test set 성능 평가 후 dog.jpg 단일 이미지 예측 수행

<details>
	<summary>전체 코드</summary>

```python
import os  # 파일 경로 처리를 위해 사용합니다.
import cv2 as cv  # 이미지 읽기, 변환, 예측 시각화를 위해 사용합니다.
import numpy as np  # 배열 처리와 확률 평균 계산을 위해 사용합니다.
import tensorflow as tf  # CIFAR-10 로드와 모델 학습을 위해 사용합니다.
from tensorflow.keras import layers, models  # Keras 레이어와 모델 API를 가져옵니다.


CLASS_NAMES = [
	"airplane",
	"automobile",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck",
]


def load_image_bgr(image_path):
	image = cv.imread(image_path)  # 일반 경로로 이미지를 먼저 읽습니다.
	if image is None:
		raw = np.fromfile(image_path, dtype=np.uint8)  # 바이트로 직접 읽어봅니다.
		if raw.size > 0:
			image = cv.imdecode(raw, cv.IMREAD_COLOR)  # OpenCV 디코딩을 시도합니다.
	return image  # 로드 결과를 반환합니다.


def preprocess_dataset(x_train, x_test):
	x_train = x_train.astype("float32") / 255.0  # 훈련 이미지를 정규화합니다.
	x_test = x_test.astype("float32") / 255.0  # 테스트 이미지도 정규화합니다.
	return x_train, x_test  # 정규화된 데이터를 반환합니다.


def make_square_crop_views(image_rgb):
	h, w = image_rgb.shape[:2]  # 원본 이미지 크기를 얻습니다.
	side = min(h, w)  # 정사각형 크롭의 한 변 길이를 정합니다.

	y_center = (h - side) // 2  # 중앙 크롭의 y 시작점을 계산합니다.
	x_center = (w - side) // 2  # 중앙 크롭의 x 시작점을 계산합니다.
	center = image_rgb[y_center : y_center + side, x_center : x_center + side]  # 중앙 크롭입니다.

	top_left = image_rgb[0:side, 0:side]  # 좌상단 크롭입니다.
	top_right = image_rgb[0:side, w - side : w]  # 우상단 크롭입니다.
	bottom_left = image_rgb[h - side : h, 0:side]  # 좌하단 크롭입니다.
	bottom_right = image_rgb[h - side : h, w - side : w]  # 우하단 크롭입니다.

	return [center, top_left, top_right, bottom_left, bottom_right]  # 5개 크롭을 반환합니다.


def build_data_augmentation():
	return models.Sequential(  # 학습 시 데이터 증강을 적용합니다.
		[
			layers.RandomFlip("horizontal"),  # 좌우 반전을 적용합니다.
			layers.RandomRotation(0.08),  # 작은 회전 변형을 줍니다.
			layers.RandomZoom(height_factor=0.1, width_factor=0.1),  # 확대/축소 변형을 줍니다.
			layers.RandomContrast(factor=0.1),  # 대비 변형을 줍니다.
		],
		name="data_augmentation",  # 증강 레이어 이름입니다.
	)


def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
	data_augmentation = build_data_augmentation()  # 증강 블록을 준비합니다.

	model = models.Sequential(  # CNN 분류 모델을 구성합니다.
		[
			layers.Input(shape=input_shape),  # 입력 크기를 정의합니다.
			data_augmentation,  # 학습 시 증강을 적용합니다.
			layers.Conv2D(32, (3, 3), activation="relu", padding="same"),  # 첫 번째 합성곱 층입니다.
			layers.BatchNormalization(),  # 배치 정규화를 적용합니다.
			layers.MaxPooling2D((2, 2)),  # 공간 크기를 줄입니다.
			layers.Conv2D(64, (3, 3), activation="relu", padding="same"),  # 두 번째 합성곱 층입니다.
			layers.BatchNormalization(),  # 배치 정규화를 적용합니다.
			layers.MaxPooling2D((2, 2)),  # 공간 크기를 다시 줄입니다.
			layers.Conv2D(128, (3, 3), activation="relu", padding="same"),  # 세 번째 합성곱 층입니다.
			layers.BatchNormalization(),  # 학습을 안정화합니다.
			layers.MaxPooling2D((2, 2)),  # 특징맵 크기를 줄입니다.
			layers.Flatten(),  # 1차원 벡터로 펼칩니다.
			layers.Dense(128, activation="relu"),  # 완전연결 은닉층입니다.
			layers.BatchNormalization(),  # 배치 정규화를 적용합니다.
			layers.Dropout(0.3),  # 과적합을 줄입니다.
			layers.Dense(num_classes, activation="softmax"),  # 10개 클래스 확률을 출력합니다.
		]
	)

	model.compile(  # 학습 설정을 컴파일합니다.
		optimizer="adam",  # Adam 최적화기를 사용합니다.
		loss="sparse_categorical_crossentropy",  # 다중분류 손실입니다.
		metrics=["accuracy"],  # 정확도를 함께 봅니다.
	)
	return model  # 컴파일된 모델을 반환합니다.


def predict_single_image(model, image_path):
	image_bgr = load_image_bgr(image_path)  # 이미지를 불러옵니다.
	if image_bgr is None:
		raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

	image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)  # BGR을 RGB로 변환합니다.
	square_views = make_square_crop_views(image_rgb)  # 여러 크롭 뷰를 만듭니다.

	batch = []  # 예측용 배치를 준비합니다.
	for view in square_views:
		resized = cv.resize(view, (32, 32), interpolation=cv.INTER_AREA)  # CIFAR-10 크기로 맞춥니다.
		batch.append(resized)  # 원본 뷰를 추가합니다.
		batch.append(cv.flip(resized, 1))  # 좌우 반전 뷰를 추가합니다.

	batch_input = np.array(batch, dtype="float32") / 255.0  # 모델 입력 형태로 변환합니다.
	probs_batch = model.predict(batch_input, verbose=0)  # 각 뷰의 확률을 예측합니다.
	probs = np.mean(probs_batch, axis=0)  # 뷰별 확률을 평균냅니다.
	pred_idx = int(np.argmax(probs))  # 가장 높은 확률의 클래스를 선택합니다.
	return pred_idx, probs  # 예측 결과를 반환합니다.


def main():
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  # CIFAR-10을 불러옵니다.
	y_train = y_train.flatten()  # 라벨을 1차원으로 변환합니다.
	y_test = y_test.flatten()  # 테스트 라벨도 1차원으로 변환합니다.

	x_train, x_test = preprocess_dataset(x_train, x_test)  # 데이터 정규화를 수행합니다.

	print(f"Train set: {x_train.shape}, labels: {y_train.shape}")  # 훈련 데이터 크기를 출력합니다.
	print(f"Test set:  {x_test.shape}, labels: {y_test.shape}")  # 테스트 데이터 크기를 출력합니다.

	model = build_cnn_model()  # CNN 모델을 생성합니다.
	model.summary()  # 모델 구조를 출력합니다.

	early_stop = tf.keras.callbacks.EarlyStopping(  # 조기 종료 콜백입니다.
		monitor="val_accuracy",  # 검증 정확도를 기준으로 봅니다.
		patience=3,  # 3회 개선이 없으면 종료합니다.
		restore_best_weights=True,  # 가장 좋은 가중치를 복원합니다.
	)

	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(  # 학습률 감소 콜백입니다.
		monitor="val_loss",  # 검증 손실을 기준으로 봅니다.
		factor=0.5,  # 학습률을 절반으로 줄입니다.
		patience=2,  # 2회 정체 시 감소합니다.
		min_lr=1e-5,  # 최소 학습률을 제한합니다.
		verbose=1,  # 변경 로그를 출력합니다.
	)

	model.fit(  # 모델 학습을 시작합니다.
		x_train,  # 입력 이미지입니다.
		y_train,  # 정답 라벨입니다.
		epochs=15,  # 최대 15회 학습합니다.
		batch_size=64,  # 배치 크기를 설정합니다.
		validation_split=0.1,  # 검증용 데이터를 분리합니다.
		callbacks=[early_stop, reduce_lr],  # 학습 안정화 콜백을 사용합니다.
		verbose=1,  # 학습 로그를 자세히 출력합니다.
	)

	test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)  # 테스트셋으로 평가합니다.
	print(f"\nTest loss: {test_loss:.4f}")  # 테스트 손실을 출력합니다.
	print(f"Test accuracy: {test_accuracy:.4f}")  # 테스트 정확도를 출력합니다.

	script_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 위치를 구합니다.
	dog_image_path = os.path.join(script_dir, "dog.jpg")  # dog.jpg 경로를 만듭니다.

	pred_idx, probs = predict_single_image(model, dog_image_path)  # 단일 이미지를 예측합니다.
	print("\nPrediction for dog.jpg")  # 예측 결과 제목을 출력합니다.
	print(f"Predicted class index: {pred_idx}")  # 예측 클래스 인덱스를 출력합니다.
	print(f"Predicted class name: {CLASS_NAMES[pred_idx]}")  # 예측 클래스 이름을 출력합니다.
	print(f"Dog probability: {probs[5]:.4f}")  # dog 클래스 확률을 출력합니다.

	top3_idx = np.argsort(probs)[-3:][::-1]  # 상위 3개 클래스를 구합니다.
	print("Top-3 probabilities:")  # 상위 3개 확률을 출력합니다.
	for idx in top3_idx:
		print(f"  {CLASS_NAMES[idx]}: {probs[idx]:.4f}")  # 각 클래스 확률을 출력합니다.


if __name__ == "__main__":
	main()
```

</details>

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

### 실행 결과

```text
Test loss: 0.7301
Test accuracy: 0.7557

Prediction for dog.jpg
Predicted class index: 5
Predicted class name: dog
Dog probability: 0.7238
Top-3 probabilities:
  dog: 0.7238
  bird: 0.1293
  cat: 0.1010
```


