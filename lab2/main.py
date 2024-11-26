import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from scipy.io import loadmat
import scipy.io

# Загрузка данных ex2data1.txt
data1 = pd.read_csv("data/ex2data1.txt", header=None, names=["Exam 1", "Exam 2", "Admitted"])
print(data1.head())

# Построение графика оценок студентов
admitted = data1[data1["Admitted"] == 1]
not_admitted = data1[data1["Admitted"] == 0]

plt.scatter(admitted["Exam 1"], admitted["Exam 2"], marker="o", label="Поступил")
plt.scatter(not_admitted["Exam 1"], not_admitted["Exam 2"], marker="x", label="Не поступил")
plt.xlabel("Оценка по первому экзамену")
plt.ylabel("Оценка по второму экзамену")
plt.title("График оценок студентов")
plt.legend()
plt.grid(True)
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Функция потерь J(θ) для логистической регрессии
def compute_cost(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))
    return cost

# Функция градиентного спуска для логистической регрессии
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        gradients = (1 / m) * X.T.dot(sigmoid(X.dot(theta)) - y)
        theta -= alpha * gradients
        cost_history.append(compute_cost(theta, X, y))
    
    return theta, cost_history

# Подготовка данных
X1 = data1[["Exam 1", "Exam 2"]].values
X1 = np.c_[np.ones(X1.shape[0]), X1]  # Добавляем столбец единичных значений для θ0
y1 = data1["Admitted"].values
theta1 = np.zeros(X1.shape[1])

# Обучение модели с использованием градиентного спуска
alpha = 0.1
iterations = 50
theta1, cost_history = gradient_descent(X1, y1, theta1, alpha, iterations)

# Функция для оптимизации с помощью метода Нелдера-Мида
def optimize_nelder_mead(theta, X, y):
    result = minimize(fun=compute_cost, x0=theta, args=(X, y), method="Nelder-Mead", options={"disp": True})
    return result.x

# Метод Бройдена-Флетчера-Гольдфарба-Шанно (BFGS)
def optimize_bfgs(theta, X, y):
    result = minimize(fun=compute_cost, x0=theta, args=(X, y), method="BFGS", options={"disp": True})
    return result.x

theta_initial = np.zeros(X1.shape[1])

# Оптимизация методом Нелдера-Мида
theta_nm = optimize_nelder_mead(theta_initial, X1, y1)
print("Оптимизированные параметры с помощью Nelder-Mead:", theta_nm)

# Оптимизация методом BFGS
theta_bfgs = optimize_bfgs(theta_nm, X1, y1)
print("Оптимизированные параметры с помощью BFGS:", theta_bfgs)

# Функция предсказания вероятности
def predict(theta, X):
    return sigmoid(X.dot(theta)) > 0.5

# Функция расчета точности
def get_accuracy(theta, X, y):
    p = predict(theta, X)
    return y[p == y].size / y.size * 100

# Рассчитываем точность на обучающей выборке
print(f'Точность на обучающей выборке: {get_accuracy(theta_bfgs, X1, y1)}%')

# Построение разделяющей прямой
x_vals = np.linspace(data1["Exam 1"].min(), data1["Exam 1"].max(), 100)
y_vals = -(theta_bfgs[0] + theta_bfgs[1] * x_vals) / theta_bfgs[2]  # Решение уравнения для вероятности 0.5

# График с разделяющей прямой
plt.scatter(admitted["Exam 1"], admitted["Exam 2"], marker="o", label="Поступил")
plt.scatter(not_admitted["Exam 1"], not_admitted["Exam 2"], marker="x", label="Не поступил")
plt.plot(x_vals, y_vals, color="red", label="Разделяющая прямая")
plt.xlabel("Оценка по первому экзамену")
plt.ylabel("Оценка по второму экзамену")
plt.title("Разделяющая прямая для логистической регрессии")
plt.legend()
plt.grid(True)
plt.show()

# Загрузка данных ex2data2.txt
data2 = pd.read_csv("data/ex2data2.txt", header=None, names=["Test 1", "Test 2", "Passed"])
print(data2.head())

# Построение графика для ex2data2.txt
passed = data2[data2["Passed"] == 1]
not_passed = data2[data2["Passed"] == 0]

plt.scatter(passed["Test 1"], passed["Test 2"], marker="o", label="Прошел контроль")
plt.scatter(not_passed["Test 1"], not_passed["Test 2"], marker="x", label="Не прошел контроль")
plt.xlabel("Результат первого теста")
plt.ylabel("Результат второго теста")
plt.title("Результаты тестов изделий")
plt.legend()
plt.grid(True)
plt.show()

# Построение полиномиальных признаков
poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(data2[["Test 1", "Test 2"]])

# L2-регуляризация для логистической регрессии
def compute_regularized_cost(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + (lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
    return cost

def gradient_descent_regularized(X, y, theta, alpha, iterations, lambda_):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        gradients = (1 / m) * X.T.dot(sigmoid(X.dot(theta)) - y)
        # Регуляризация
        gradients[1:] += (lambda_ / m) * theta[1:]
        theta -= alpha * gradients
        cost_history.append(compute_regularized_cost(theta, X, y, lambda_))
    
    return theta, cost_history

# Регуляризация
lambda_ = 1
theta_reg = np.zeros(X_poly.shape[1])
theta_reg, cost_history_reg = gradient_descent_regularized(X_poly, data2["Passed"].values, theta_reg, 0.1, 400, lambda_)

# Оптимизация методом BFGS
theta_reg_bfgs = optimize_bfgs(np.zeros(X_poly.shape[1]), X_poly, data2["Passed"].values)
print("Оптимизированные параметры с помощью BFGS:", theta_reg_bfgs)

theta_reg_nelder_mead = optimize_nelder_mead(np.zeros(X_poly.shape[1]), X_poly, data2["Passed"].values)
print("Оптимизированные параметры с помощью метода Нелдера-Мида:", theta_reg_nelder_mead)

# Предсказание вероятности прохождения контроля
def predict_probabilities_control(theta, X):
    return sigmoid(X.dot(theta))

# Предсказания
pred_probs_control = predict_probabilities_control(theta_reg_bfgs, X_poly)

xx, yy = np.meshgrid(np.linspace(data2["Test 1"].min(), data2["Test 1"].max(), 100),
                     np.linspace(data2["Test 2"].min(), data2["Test 2"].max(), 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]  
grid_points_poly = poly.transform(grid_points)
Z = predict_probabilities_control(theta_reg_bfgs, grid_points_poly)

# Функция предсказания вероятности
def predict(theta, X):
    return sigmoid(X.dot(theta)) > 0.5

# Функция расчета точности
def get_accuracy(theta, X, y):
    p = predict(theta, X)
    return y[p == y].size / y.size * 100

y_acc = np.array(data2['Passed'])
# Рассчитываем точность на обучающей выборке
print(f'Точность на второй обучающей выборке: {get_accuracy(theta_reg_bfgs, X_poly, y_acc)}%')

# Преобразуем предсказания обратно в нужную форму (100, 100)
Z = Z.reshape(xx.shape)

# Построение разделяющей кривой
plt.contour(xx, yy, Z, levels=[0.5], cmap="viridis")  # Уровень 0.5 для разделяющей кривой
plt.scatter(data2["Test 1"], data2["Test 2"], c=data2["Passed"], cmap="coolwarm", edgecolors='k')
plt.xlabel("Test 1")
plt.ylabel("Test 2")
plt.title("Разделяющая кривая для вероятности прохождения контроля")
plt.show()

# Влияние значения регуляризации λ
lambda_values = [0.01, 0.1, 1, 10]

for lambda_ in lambda_values:
    theta_reg = np.zeros(X_poly.shape[1])
    theta_reg, _ = gradient_descent_regularized(X_poly, data2["Passed"].values, theta_reg, 0.1, 400, lambda_)

    xx, yy = np.meshgrid(np.linspace(data2["Test 1"].min(), data2["Test 1"].max(), 100),
                         np.linspace(data2["Test 2"].min(), data2["Test 2"].max(), 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_poly = poly.transform(grid_points)  # Преобразование в полиномиальные признаки
    
    # Предсказание вероятностей для точек сетки
    Z = predict_probabilities_control(theta_reg, grid_points_poly)
    Z = Z.reshape(xx.shape)  # Преобразование в форму сетки
    
    plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors="blue", label=f"λ = {lambda_}")
    plt.scatter(passed["Test 1"], passed["Test 2"], marker="o", label="Прошел контроль")
    plt.scatter(not_passed["Test 1"], not_passed["Test 2"], marker="x", label="Не прошел контроль")
    plt.xlabel("Результат первого теста")
    plt.ylabel("Результат второго теста")
    plt.title(f"Разделяющая кривая для λ = {lambda_}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Загрузка данных ex2data3.mat
data = scipy.io.loadmat("data/ex2data3.mat")
X = data["X"]  # Матрица изображений размерностью (5000, 400)
y = data["y"]  # Метки классов размерностью (5000, 1)

# Преобразуем метки класса: 10 -> 0
y = y.flatten()
y[y == 10] = 0

# Визуализация случайных изображений
def plot_random_samples(X, y, samples_per_class=1):
    """Визуализация нескольких изображений из набора данных."""
    classes = np.unique(y)
    fig, axes = plt.subplots(samples_per_class, len(classes), figsize=(10, 10))

    for i, cls in enumerate(classes):
        indices = np.where(y == cls)[0]
        chosen_indices = np.random.choice(indices, samples_per_class, replace=False)

        for j, idx in enumerate(chosen_indices):
            ax = axes[j, i] if samples_per_class > 1 else axes[i]
            ax.imshow(X[idx].reshape(20, 20).T, cmap="gray")
            ax.axis("off")
            if j == 0:
                ax.set_title(f"Цифра: {cls}")

    plt.tight_layout()
    plt.show()

plot_random_samples(X, y, samples_per_class=1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_reg(theta, X, y, lambda_):
    m = len(y)
    h = sigmoid(X.dot(theta))
    reg_term = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    cost = (-1 / m) * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h))) + reg_term
    return cost

def gradient_descent_reg(X, y, theta, alpha, iterations, lambda_):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        h = sigmoid(X.dot(theta))
        reg_term = (lambda_ / m) * np.r_[[0], theta[1:]]  # Регуляризация
        gradients = (1 / m) * X.T.dot(h - y) + reg_term
        theta -= alpha * gradients
        cost_history.append(compute_cost_reg(theta, X, y, lambda_))
    
    return theta, cost_history

# Функция многоклассовой классификация, метод "один против всех"
def one_vs_all(X, y, num_labels, alpha, iterations, lambda_):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n))

    for c in range(num_labels):
        initial_theta = np.zeros(n)
        y_c = (y == c).astype(int)
        theta, _ = gradient_descent_reg(X, y_c, initial_theta, alpha, iterations, lambda_)
        all_theta[c] = theta

        #print(f"Класс {c}: Вес = {_[-1]:.4f}")

    return all_theta

# Функция предсказания класса
def predict_one_vs_all(all_theta, X):
    probs = sigmoid(X.dot(all_theta.T))
    return np.argmax(probs, axis=1)

X_with_bias = np.c_[np.ones(X.shape[0]), X]

split_ratio = 0.8
num_samples = X_with_bias.shape[0]

# Количество примеров в тренировочной выборке
train_size = int(num_samples * split_ratio)

# Перемешивание индексов
indices = np.random.permutation(num_samples)

# Разделение на тренировочную и тестовую выборки
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X_with_bias[train_indices]
X_test = X_with_bias[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

# Параметры обучения
alpha = 0.1
iterations = 7000
lambda_ = 0.1
num_labels = 10

# Обучение
all_theta = one_vs_all(X_train, y_train, num_labels, alpha, iterations, lambda_)

# Предсказание
y_pred = predict_one_vs_all(all_theta, X_test)

# Оценка точности
accuracy = np.mean(y_pred == y_test) * 100
print(f"Точность классификации: {accuracy:.2f}%")