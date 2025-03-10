import hashlib
import tarfile
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def load_data(archive_path):
    images = []
    labels = []
    class_names = []  
    valid_extensions = ['.png']  

    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue  

            folder_name = member.name.split('/')[1]  
            if folder_name not in class_names:
                class_names.append(folder_name)  

            label = class_names.index(folder_name)  

            if any(member.name.endswith(ext) for ext in valid_extensions):
                file_obj = archive.extractfile(member)
                try:
                    img = Image.open(BytesIO(file_obj.read()))
                    img = img.convert('L') 
                    img_array = np.array(img)

                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Ошибка при обработке файла {member.name}: {e}")
            else:
                print(f"Пропущен файл {member.name} с неподдерживаемым расширением")

    return np.array(images), np.array(labels), class_names

def check_class_balance(labels, class_names):
    class_counts = np.zeros(len(class_names), dtype=int)

    for label in labels:
        class_counts[label] += 1

    for i, class_name in enumerate(class_names):
        print(f"Класс {class_name}: {class_counts[i]} изображений")

def display_random_images(images, labels, class_names, num_images=10):
    random_indices = np.random.choice(len(images), num_images, replace=False)

    plt.figure(figsize=(10, 5))
    
    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)  
        plt.imshow(images[idx], cmap='gray')  
        plt.title(f"Class: {class_names[labels[idx]]}") 
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def check_no_duplicates(X_train, X_val, X_test, y_train, y_val, y_test):
    X_train_flat = [x.tobytes() for x in X_train]  
    X_val_flat = [x.tobytes() for x in X_val]
    X_test_flat = [x.tobytes() for x in X_test]

    train_val_overlap = np.intersect1d(X_train_flat, X_val_flat)
    train_test_overlap = np.intersect1d(X_train_flat, X_test_flat)

    if len(train_val_overlap) > 0:
        print(f"Найдены дубликаты между обучающей и валидационной выборками: {len(train_val_overlap)}")
    if len(train_test_overlap) > 0:
        print(f"Найдены дубликаты между обучающей и тестовой выборками: {len(train_test_overlap)}")
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0:
        print("Нет дубликатов между обучающей выборкой и остальными выборками.")

def hash_image(image):
    return hashlib.sha256(image.tobytes()).hexdigest()

def remove_duplicates_h(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train_hashes = [hash_image(x) for x in X_train]
    X_val_hashes = set(hash_image(x) for x in X_val)  
    X_test_hashes = set(hash_image(x) for x in X_test)

    overlap_indices = []
    for i, hash_val in enumerate(X_train_hashes):
        if hash_val in X_val_hashes or hash_val in X_test_hashes:
            overlap_indices.append(i)

    X_train_cleaned = np.delete(X_train, overlap_indices, axis=0)
    y_train_cleaned = np.delete(y_train, overlap_indices)

    return X_train_cleaned, y_train_cleaned

if __name__ == "__main__":
    print("Функции из 1 ЛР")