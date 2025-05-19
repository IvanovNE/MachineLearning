import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # нормализация
    return image, label

# Загрузка и предобработка данных MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Преобразование размеров изображений и нормализация
train_images = train_images.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
test_images = test_images.reshape((-1, 28, 28, 1)).astype("float32") / 255.0

# Построение модели CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 классов
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

# Оценка модели
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nТочность на тестовых данных: {test_acc:.4f}")


(ds_train, ds_test), ds_info = tfds.load(
    'svhn_cropped',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

batch_size = 64

ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

# Переобучение той же модели на данных SVHN
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_svhn = model.fit(ds_train, epochs=5, validation_data=ds_test)

# Оценка модели на тестовом наборе SVHN
test_loss_svhn, test_acc_svhn = model.evaluate(ds_test, verbose=2)
print(f"\nТочность на тестовых данных SVHN: {test_acc_svhn:.4f}")