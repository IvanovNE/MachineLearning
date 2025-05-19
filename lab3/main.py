import lab2f
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Rescaling, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Flatten

def create_first_model(input_shape, num_classes):

    model = Sequential()

    model.add(Rescaling(scale=1/.255, input_shape=input_shape))
    model.add(Conv2D(16, (3,3), activation = 'relu'))
    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(64, input_shape=input_shape, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))
    return model

def create_second_model(input_shape, num_classes):

    model = Sequential()

    model.add(Rescaling(scale=1/.255, input_shape=input_shape))
    model.add(Conv2D(16, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(64, input_shape=input_shape, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))
    return model

def create_third_model(input_shape, num_classes):

    model = Sequential()

    model.add(Rescaling(scale=1./255, input_shape=input_shape))
    model.add(Conv2D(32, (5,5), activation='tanh', padding='same'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5,5), activation='tanh', padding='valid')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='tanh'))  
    model.add(Dense(84, activation='tanh'))   

    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model(model, X_train, y_train, X_val, y_val, initial_lr, epochs, batch_size):

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
        verbose=1
    )
    return history

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return loss, accuracy

################################################################################################

# archive_path = 'notMNIST_small.tar.gz'
archive_path = 'notMNIST_large.tar.gz'

images, labels, class_names = lab2f.load_data(archive_path)

X_train, y_train, X_val, y_val, X_test, y_test = lab2f.prepare_data(images, labels, class_names)

X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

input_shape = (28, 28, 1)
num_classes = len(np.unique(y_train))

model = create_first_model(input_shape, num_classes)

history = train_model(model, X_train, y_train, X_val, y_val, 0.1, 10, 64)

lab2f.create_plot(history)
     
loss, accuracy = evaluate_model(model, X_test, y_test)

print()
print("Точность 1й модели: " + str(accuracy) + "; потери: " + str(loss))
print()

model = create_second_model(input_shape, num_classes)

history = train_model(model, X_train, y_train, X_val, y_val, 0.1, 10, 64)

lab2f.create_plot(history)
     
loss, accuracy = evaluate_model(model, X_test, y_test)

print()
print("Точность 2й модели: " + str(accuracy) + "; потери: " + str(loss))
print()

X_train = X_train.astype('float32') * 255.0
X_val = X_val.astype('float32') * 255.0
X_test = X_test.astype('float32') * 255.0

model = create_third_model(input_shape, num_classes)

history = train_model(model, X_train, y_train, X_val, y_val, 0.01, 10, 64)

lab2f.create_plot(history)
     
loss, accuracy = evaluate_model(model, X_test, y_test)

print()
print("Точность 3й модели: " + str(accuracy) + "; потери: " + str(loss))
print()