import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Функция потерь для линейной регрессии
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors**2)
    return cost

# Функция градиентного спуска
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        gradients = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradients
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

# Загрузка данных ex1data1.txt
data1 = pd.read_csv("data/ex1data1.txt", header=None, names=["Population", "Profit"])
print(data1.head())

# Загрузка данных ex1data2.txt
data2 = pd.read_csv("data/ex1data2.txt", header=None, names=["Size", "Rooms", "Price"])
print(data2.head())

# Подготовка данных
data1.insert(0, "Intercept", 1)  # Добавляем столбец для theta0
X = data1[["Intercept", "Population"]].values
y = data1["Profit"].values
theta = np.zeros(2)

# Градиентный спуск
alpha = 0.01
iterations = 1500
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

# График 1
plt.scatter(data1["Population"], data1["Profit"], c='green', marker='x')
# Построение модели
plt.plot(data1["Population"], X.dot(theta), color='red')

plt.legend(["Прибыль от населения", "Градиентный спуск"], loc="lower right")
plt.xlabel("Население города")
plt.ylabel("Прибыль")
plt.title("Зависимость прибыли ресторана от населения города")
plt.grid(True)
plt.show()


theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i, t0 in enumerate(theta0_vals):
    for j, t1 in enumerate(theta1_vals):
        t = np.array([t0, t1])
        J_vals[i, j] = compute_cost(X, y, t)

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

# 3D Surface Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(theta0_vals, theta1_vals, J_vals.T, cmap="viridis")
ax.set_xlabel("θ0")
ax.set_ylabel("θ1")
ax.set_zlabel("Потери")
plt.title("График зависимости функции потерь от параметров модели в виде поверхности")
plt.show()

# Contour Plot
plt.contour(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-2, 3, 20), cmap="viridis")
plt.xlabel("θ0")
plt.ylabel("θ1")
plt.title("График зависимости функции потерь от параметров модели в виде изолиний")
plt.plot(theta[0], theta[1], 'rx', markersize=10, label="Оптимизированные параметры модели")
plt.legend()
plt.show()

# Функция нормализации признаков
def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# Нормализация для ex1data2.txt
X2 = data2[["Size", "Rooms"]].values
y2 = data2["Price"].values
X2, mu, sigma = feature_normalize(X2)

# Функция для подготовки данных
def prepare_features(data, features, target, normalize=True):
    X = data[features].values
    y = data[target].values
    
    if normalize:
        X, mu, sigma = feature_normalize(X)
    else:
        mu, sigma = None, None
        
    # Добавляем столбец единиц для theta0
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    return X, y, mu, sigma

# Функция градиентного спуска с поддержкой векторизации
def gradient_descent_vectorized(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        gradients = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradients
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

# Подготовка данных для нормализованной и ненормализованной модели
features = ["Size", "Rooms"]
target = "Price"

# Нормализованные данные
X_norm, y_norm, mu_norm, sigma_norm = prepare_features(data2, features, target, normalize=True)

# Ненормализованные данные
X_nonorm, y_nonorm, _, _ = prepare_features(data2, features, target, normalize=False)

# Начальные параметры
theta_norm = np.zeros(X_norm.shape[1])
theta_nonorm = np.zeros(X_nonorm.shape[1])

# Параметры градиентного спуска
alpha = 0.1
alpha1 = 0.00000001
iterations = 50

# Выполнение градиентного спуска
theta_norm, cost_history_norm = gradient_descent_vectorized(X_norm, y_norm, theta_norm, alpha, iterations)
theta_nonorm, cost_history_nonorm = gradient_descent_vectorized(X_nonorm, y_nonorm, theta_nonorm, alpha1, iterations)

# Сравнение графиков сходимости
plt.plot(range(len(cost_history_norm)), cost_history_norm, label="Нормализованные признаки")
plt.plot(range(len(cost_history_nonorm)), cost_history_nonorm, label="Ненормализованные признаки")
plt.xlabel("Итерации")
plt.ylabel("Значение функции потерь")
plt.title("Сравнение сходимости")
plt.legend()
plt.grid(True)
plt.show()

# Функция изменения коэффициента обучения α
def compare_learning_rates(X, y, theta, alphas, iterations):
    for iteration in iterations:
        for alpha in alphas:
            theta_temp = theta.copy()
            _, cost_history = gradient_descent_vectorized(X, y, theta_temp, alpha, iteration)
            plt.plot(range(len(cost_history)), cost_history, label=f"α = {alpha:.3f}")

        plt.xlabel("Итерации")
        plt.ylabel("Значение функции потерь")
        plt.title(f"Влияние коэффициента обучения на сходимость ({iteration} итераций)")
        plt.legend()
        plt.grid(True)
        plt.show()

# Сравнение с различными α
alphas = np.linspace(0.001, 0.1, 6).tolist()
iteration_list = np.linspace(50, 150, 3, dtype=int).tolist()
compare_learning_rates(X_norm, y_norm, np.zeros(X_norm.shape[1]), alphas, iteration_list)

# Функция метода наименьших квадратов
def normal_equation(X, y):
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

# Аналитическое решение
theta_analytical = normal_equation(X_nonorm, y_nonorm)

# Сравнение результатов
print("Параметры методом градиентного спуска (нормализованные):", theta_norm)
print("Параметры аналитическим решением:", theta_analytical)

# Функция оценки производительности модели
def measure_performance(X, y, theta, alpha, iterations, vectorized=True):
    start_time = time.time()
    if vectorized:
        _, _ = gradient_descent_vectorized(X, y, theta, alpha, iterations)
    else:
        _, _ = gradient_descent(X, y, theta, alpha, iterations)
    elapsed_time = time.time() - start_time
    return elapsed_time

# Измерение времени
time_vectorized = measure_performance(X_norm, y_norm, np.zeros(X_norm.shape[1]), alpha, iterations, vectorized=True)
time_nonvectorized = measure_performance(X_norm, y_norm, np.zeros(X_norm.shape[1]), alpha, iterations, vectorized=False)

print(f"Время (с веторизацией): {time_vectorized:.4f} секунд")
print(f"Время (без векторизации): {time_nonvectorized:.4f} секунд")