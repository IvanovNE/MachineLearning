import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Универсальная предобработка для SVHN: grayscale + resize до 28x28 + нормализация
def preprocess_svhn(image, label):
    image = tf.image.rgb_to_grayscale(image)  # 3 -> 1 канал
    image = tf.image.resize(image, [28, 28])   # 32x32 -> 28x28
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Загружаем MNIST и предобрабатываем
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
test_images = test_images.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

# Модель CNN, подходящая для 28x28x1
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Компиляция и обучение на MNIST
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nТочность на тестовых данных MNIST: {test_acc:.4f}")

# Загрузка SVHN и применение преобразований
(ds_train, ds_test), ds_info = tfds.load(
    'svhn_cropped',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

batch_size = 64

ds_train = ds_train.map(preprocess_svhn, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(preprocess_svhn, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

# Повторное обучение модели на SVHN
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_svhn = model.fit(ds_train, epochs=20, validation_data=ds_test)

test_loss_svhn, test_acc_svhn = model.evaluate(ds_test, verbose=2)
print(f"\nТочность на тестовых данных SVHN: {test_acc_svhn:.4f}")

model.save("housenumber_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("housenumber_model.tflite", "wb") as f:
    f.write(tflite_model)