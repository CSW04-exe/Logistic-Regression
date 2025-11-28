# ----------------------------------------------------------------
# Author : Carter Ward
# Class  : CS 430-1
# Date   : 10/25/2025
#
# Purpose: This program performs logistic regression for binary classification of Iris-setosa vs. other species.
#          1) Loads feature and label data from "iris.data"
#          2) Preprocesses data by adding intercept term
#          3) Splits the dataset into training and validation sets (80/20)
#          4) Implements sigmoid activation, cost function, and its gradient manually
#          5) Optimizes parameters using batch gradient descent
#          6) Predicts labels on validation set and evaluates with confusion matrix
#          7) Calculates and prints optimal parameters, accuracy, and precision
# ----------------------------------------------------------------

import random
import math

# Implements the sigmoid activation function
def sigmoid(z):
    if isinstance(z, list):
        return [1 / (1 + math.exp(-x)) for x in z]
    return 1 / (1 + math.exp(-z))

# Loads the iris dataset and maps class labels to binary (Setosa vs Others)
def load_iris_data(file_path: str):
    data = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            if len(row) != 5:
                continue
            features = list(map(float, row[:4]))
            label = 1 if row[4] == 'Iris-setosa' else 0  # Binary: Setosa vs Others
            data.append(features)
            labels.append(label)
    return data, labels

# Adds intercept term for bias
def add_intercept(X):
    return [[1.0] + row for row in X]

# Computes dot product of two vectors
def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

# Computes the logistic regression cost function
def compute_cost(theta, X, y):
    m = len(y)
    cost = 0.0
    for i in range(m):
        z = dot(theta, X[i])
        h = sigmoid(z)
        cost += -y[i] * math.log(h) - (1 - y[i]) * math.log(1 - h)
    return cost / m

# Computes the gradient of the logistic regression cost function
def compute_gradient(theta, X, y):
    m = len(y)
    grad = [0.0] * len(theta)
    for i in range(m):
        error = sigmoid(dot(theta, X[i])) - y[i]
        for j in range(len(theta)):
            grad[j] += error * X[i][j]
    return [g / m for g in grad]

# Performs gradient descent to optimize theta
def gradient_descent(X, y, theta, alpha, num_iters):
    for _ in range(num_iters):
        grad = compute_gradient(theta, X, y)
        theta = [t - alpha * g for t, g in zip(theta, grad)]
    return theta

# Predicts binary class from learned theta
def predict(theta, X):
    return [1 if sigmoid(dot(theta, x)) >= 0.5 else 0 for x in X]

# Computes the accuracy of predictions
def accuracy(y_true, y_pred):
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)

# Computes the precision of predictions
def precision(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yp == 1 and yt != yp)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

# Computes the confusion matrix from predictions
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn

# Randomly splits the data into training and validation sets
def train_test_split(X, y, test_ratio=0.2):
    combined = list(zip(X, y))
    random.shuffle(combined)
    X[:], y[:] = zip(*combined)
    split_index = int(len(y) * (1 - test_ratio))
    return list(X[:split_index]), list(X[split_index:]), list(y[:split_index]), list(y[split_index:])

# Main function to train, evaluate, and report results
def main():
    random.seed(430)
    X, y = load_iris_data("iris.data")
    X = add_intercept(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y)

    initial_theta = [0.0] * len(X[0])
    alpha = 0.1
    num_iters = 1000

    theta_opt = gradient_descent(X_train, y_train, initial_theta, alpha, num_iters)

    y_pred = predict(theta_opt, X_val)
    acc = accuracy(y_val, y_pred)
    prec = precision(y_val, y_pred)
    tp, tn, fp, fn = confusion_matrix(y_val, y_pred)

    print("Optimal theta:", theta_opt)
    print(f"Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")

if __name__ == '__main__':
    main()
