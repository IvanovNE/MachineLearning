import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

# Путь к папке с изображениями
image_folder = "./test_photo"

# Загрузка модели
model = tf.keras.models.load_model("housenumber_model.h5")

# Функция предобработки и распознавания одной цифры
def preprocess_and_predict(roi):
    roi = cv2.resize(roi, (28, 28))
    roi = roi.astype("float32") / 255.0
    roi = roi.reshape(1, 28, 28, 1)
    prediction = model.predict(roi, verbose=0)
    return np.argmax(prediction)

# Обработка всех изображений
for filename in os.listdir(image_folder):
    if filename.lower().endswith(".jpg"):
        filepath = os.path.join(image_folder, filename)

        # Загрузка изображения в градациях серого
        original = cv2.imread(filepath)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # Размытие и порог
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 11, 3
        )

        # Поиск контуров
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        digit_predictions = []
        bounding_boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Фильтрация мелких шумов и больших пятен
            if 10 < w < 100 and 10 < h < 100:
                bounding_boxes.append((x, y, w, h))

        # Сортировка контуров по x-координате (слева направо)
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

        for (x, y, w, h) in bounding_boxes:
            roi = thresh[y:y+h, x:x+w]
            predicted_digit = preprocess_and_predict(roi)
            digit_predictions.append(str(predicted_digit))

            # Отображение найденной цифры (по желанию)
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(original, str(predicted_digit), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        result = ''.join(digit_predictions)
        print(f"{filename}: распознано -> {result}")

        # Показ результата
        plt.figure(figsize=(6, 4))
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title(f"Распознано: {result}")
        plt.axis("off")
        plt.show()
