import lab1f
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Flatten

def create_model(input_shape, hidden_layers, activations, dropout_rate, num_classes):

    model = Sequential()

    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(hidden_layers[0], input_shape=input_shape, activation=activations[0]))
    model.add(Dropout(dropout_rate))

    for neurons, activation in zip(hidden_layers[1:], activations[1:]):
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))

    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model(model, X_train, y_train, X_val, y_val, initial_lr, epochs, batch_size):

    def scheduler(epoch, lr):
        return lr * 0.5 if epoch > 5 else lr
    
    lr_callback = LearningRateScheduler(scheduler)
    
    model.compile(
        optimizer=SGD(learning_rate=initial_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[lr_callback],
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test):
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

################################################################################################

archive_path = 'notMNIST_small.tar.gz'
# archive_path = 'notMNIST_large.tar.gz'

images, labels, class_names = lab1f.load_data(archive_path)

print()
print(f"Количество изображений: {images.shape[0]}")
print(f"Размер изображений: {images.shape[1:]}")
print(f"Количество классов: {len(class_names)}")
print(f"Имена классов: {class_names}")
print()

lab1f.check_class_balance(labels, class_names)

lab1f.display_random_images(images, labels, class_names)

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.15, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42)

print()
print(f"Размер обучающей выборки: {X_train.shape[0]}")
print(f"Размер валидационной выборки: {X_val.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")
print()

print(f"Балансировка обучающей выборки:")
lab1f.check_class_balance(y_train, class_names)
print()
print(f"Балансировка валидационной выборки:")
lab1f.check_class_balance(y_val, class_names)
print()
print(f"Балансировка тестовой выборки:")
lab1f.check_class_balance(y_test, class_names)
print()

lab1f.check_no_duplicates(X_train, X_val, X_test, y_train, y_val, y_test)
print()

X_train_cleaned, y_train_cleaned = lab1f.remove_duplicates_h(X_train, y_train, X_val, y_val, X_test, y_test)
print("Очистка дубликатов в обучающей выборке")
print()

lab1f.check_no_duplicates(X_train_cleaned, X_val, X_test, y_train_cleaned, y_val, y_test)
print()

hidden_layers = [1024, 512, 256] 
activations = ['relu', 'relu', 'relu', 'relu', 'sigmoid'] 
dropout_rate = 0.05  
initial_lr = 0.3
epochs = 10
batch_size = 64

X_train_cleaned = X_train_cleaned.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

input_shape = (28, 28, 1)
num_classes = len(np.unique(y_train_cleaned))

model = create_model(input_shape, hidden_layers, activations, dropout_rate, num_classes)

train_model(model, X_train_cleaned, y_train_cleaned, X_val, y_val, initial_lr, epochs, batch_size)

accuracy = evaluate_model(model, X_test, y_test)

print()
print("Точность модели: " + str(accuracy))