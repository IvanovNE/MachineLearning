import os
import matplotlib.pyplot as plt
import random
from PIL import Image

# Путь к папке с данными
data_dir = "notMNIST_small"

# Задание 1: Отображение нескольких изображений
def display_images(data_dir):
    classes = sorted(os.listdir(data_dir))  # Классы (папки A-J)
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))  # Сетка 2x5 для отображения
    axs = axs.ravel()

    for i, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, cls)
        img_name = random.choice(os.listdir(cls_path))  # Берем случайное изображение
        img_path = os.path.join(cls_path, img_name)
        img = Image.open(img_path)
        axs[i].imshow(img, cmap='gray')
        axs[i].set_title(cls)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

display_images(data_dir)

def check_class_balance(data_dir):
    classes = sorted(os.listdir(data_dir))
    counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}
    
    # Отображение количества изображений по классам
    for cls, count in counts.items():
        print(f"Class {cls}: {count} images")
    
    # График распределения
    plt.bar(counts.keys(), counts.values())
    plt.xlabel("Class")
    plt.ylabel("Number of images")
    plt.title("Class Balance")
    plt.show()

check_class_balance(data_dir)
