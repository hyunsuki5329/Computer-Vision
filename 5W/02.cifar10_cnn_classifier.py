import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


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
    image = cv.imread(image_path)
    if image is None:
        raw = np.fromfile(image_path, dtype=np.uint8)
        if raw.size > 0:
            image = cv.imdecode(raw, cv.IMREAD_COLOR)
    return image


def preprocess_dataset(x_train, x_test):
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, x_test


def make_square_crop_views(image_rgb):
    h, w = image_rgb.shape[:2]
    side = min(h, w)

    y_center = (h - side) // 2
    x_center = (w - side) // 2
    center = image_rgb[y_center : y_center + side, x_center : x_center + side]

    top_left = image_rgb[0:side, 0:side]
    top_right = image_rgb[0:side, w - side : w]
    bottom_left = image_rgb[h - side : h, 0:side]
    bottom_right = image_rgb[h - side : h, w - side : w]

    return [center, top_left, top_right, bottom_left, bottom_right]


def build_data_augmentation():
    return models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(height_factor=0.1, width_factor=0.1),
            layers.RandomContrast(factor=0.1),
        ],
        name="data_augmentation",
    )


def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    data_augmentation = build_data_augmentation()

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            data_augmentation,
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_single_image(model, image_path):
    image_bgr = load_image_bgr(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    square_views = make_square_crop_views(image_rgb)

    batch = []
    for view in square_views:
        resized = cv.resize(view, (32, 32), interpolation=cv.INTER_AREA)
        batch.append(resized)
        batch.append(cv.flip(resized, 1))

    batch_input = np.array(batch, dtype="float32") / 255.0
    probs_batch = model.predict(batch_input, verbose=0)
    probs = np.mean(probs_batch, axis=0)
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    x_train, x_test = preprocess_dataset(x_train, x_test)

    print(f"Train set: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test set:  {x_test.shape}, labels: {y_test.shape}")

    model = build_cnn_model()
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1,
    )

    model.fit(
        x_train,
        y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dog_image_path = os.path.join(script_dir, "dog.jpg")

    pred_idx, probs = predict_single_image(model, dog_image_path)
    print("\nPrediction for dog.jpg")
    print(f"Predicted class index: {pred_idx}")
    print(f"Predicted class name: {CLASS_NAMES[pred_idx]}")
    print(f"Dog probability: {probs[5]:.4f}")

    top3_idx = np.argsort(probs)[-3:][::-1]
    print("Top-3 probabilities:")
    for idx in top3_idx:
        print(f"  {CLASS_NAMES[idx]}: {probs[idx]:.4f}")


if __name__ == "__main__":
    main()
