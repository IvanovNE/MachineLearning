import lab1f
from sklearn.model_selection import train_test_split

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

sizes = [50, 100, 1000, 15000]

