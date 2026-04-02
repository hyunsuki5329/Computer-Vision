import numpy as np  # 배열 연산과 데이터 결합/셔플을 위해 NumPy를 사용합니다.
import tensorflow as tf  # 데이터셋 로드와 신경망 학습을 위해 TensorFlow를 사용합니다.
from tensorflow.keras import layers, models  # Keras의 레이어와 모델 API를 가져옵니다.


def split_dataset(images, labels, train_ratio=0.8, seed=42):
    rng = np.random.default_rng(seed)  # 재현 가능한 셔플을 위해 고정 시드 난수 생성기를 만듭니다.
    indices = rng.permutation(len(images))  # 전체 샘플 인덱스를 무작위 순서로 섞습니다.
    split_idx = int(len(images) * train_ratio)  # 훈련 세트와 테스트 세트를 나눌 기준 위치를 계산합니다.

    train_idx = indices[:split_idx]  # 앞부분 인덱스를 훈련 데이터용으로 사용합니다.
    test_idx = indices[split_idx:]  # 뒷부분 인덱스를 테스트 데이터용으로 사용합니다.

    x_train = images[train_idx]  # 훈련 이미지 데이터를 인덱스로 추출합니다.
    y_train = labels[train_idx]  # 훈련 정답 라벨을 인덱스로 추출합니다.
    x_test = images[test_idx]  # 테스트 이미지 데이터를 인덱스로 추출합니다.
    y_test = labels[test_idx]  # 테스트 정답 라벨을 인덱스로 추출합니다.
    return x_train, x_test, y_train, y_test  # 분할된 데이터 4개를 반환합니다.


def preprocess_images(images):
    return images.astype("float32") / 255.0  # 픽셀값을 0~1 범위로 정규화해 학습 안정성을 높입니다.


def build_model():
    model = models.Sequential(  # 층을 순서대로 쌓는 Sequential 모델을 생성합니다.
        [
            layers.Flatten(input_shape=(28, 28)),  # 28x28 이미지를 1차원 벡터(784)로 펼칩니다.
            layers.Dense(128, activation="relu"),  # 첫 번째 은닉층(뉴런 128개)으로 비선형 특징을 학습합니다.
            layers.Dense(64, activation="relu"),  # 두 번째 은닉층(뉴런 64개)으로 표현을 압축합니다.
            layers.Dense(10, activation="softmax"),  # 10개 숫자 클래스 확률을 출력합니다.
        ]
    )

    model.compile(  # 학습에 사용할 최적화/손실/평가 지표를 설정합니다.
        optimizer="adam",  # 적응형 학습률을 사용하는 Adam 옵티마이저를 사용합니다.
        loss="sparse_categorical_crossentropy",  # 정답이 정수 라벨일 때 쓰는 다중분류 손실입니다.
        metrics=["accuracy"],  # 학습 중 정확도를 함께 모니터링합니다.
    )
    return model  # 컴파일된 모델 객체를 반환합니다.


def main():
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()  # MNIST 기본 train/test 데이터를 로드합니다.

    all_images = np.concatenate([x_train_raw, x_test_raw], axis=0)  # 전체 이미지를 하나로 합쳐 직접 분할할 수 있게 만듭니다.
    all_labels = np.concatenate([y_train_raw, y_test_raw], axis=0)  # 전체 라벨도 같은 순서로 하나로 합칩니다.

    x_train, x_test, y_train, y_test = split_dataset(all_images, all_labels, train_ratio=0.8, seed=42)  # 전체 데이터를 8:2로 셔플 분할합니다.

    x_train = preprocess_images(x_train)  # 훈련 이미지를 정규화합니다.
    x_test = preprocess_images(x_test)  # 테스트 이미지도 같은 방식으로 정규화합니다.

    print(f"Train set: {x_train.shape}, labels: {y_train.shape}")  # 분할 결과의 훈련 데이터 크기를 출력합니다.
    print(f"Test set:  {x_test.shape}, labels: {y_test.shape}")  # 분할 결과의 테스트 데이터 크기를 출력합니다.

    model = build_model()  # 신경망 구조를 생성하고 컴파일합니다.
    model.summary()  # 모델 층별 파라미터 정보를 콘솔에 출력합니다.

    model.fit(  # 모델 학습을 시작합니다.
        x_train,  # 입력 훈련 이미지 데이터입니다.
        y_train,  # 입력 훈련 정답 라벨입니다.
        epochs=5,  # 전체 훈련 데이터를 5번 반복 학습합니다.
        batch_size=128,  # 한 번의 가중치 갱신에 사용할 샘플 수입니다.
        validation_split=0.1,  # 훈련 데이터의 10%를 검증용으로 분리합니다.
        verbose=1,  # 학습 진행 로그를 자세히 출력합니다.
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)  # 학습이 끝난 모델을 테스트 세트로 평가합니다.
    print(f"\nTest loss: {test_loss:.4f}")  # 테스트 손실 값을 소수점 4자리로 출력합니다.
    print(f"Test accuracy: {test_accuracy:.4f}")  # 테스트 정확도를 소수점 4자리로 출력합니다.

    probabilities = model.predict(x_test, verbose=0)  # 각 테스트 샘플의 클래스별 확률을 예측합니다.
    predictions = np.argmax(probabilities, axis=1)  # 가장 확률이 높은 클래스를 최종 예측값으로 선택합니다.
    confusion = tf.math.confusion_matrix(y_test, predictions, num_classes=10).numpy()  # 10개 클래스 혼동행렬을 계산합니다.

    print("\nConfusion matrix (rows=true label, cols=predicted label):")  # 혼동행렬의 행/열 의미를 안내합니다.
    print(confusion)  # 계산된 혼동행렬 숫자를 출력합니다.


if __name__ == "__main__":
    main()
