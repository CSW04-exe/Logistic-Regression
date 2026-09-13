# Logistic Regression on the Iris Dataset

A from-scratch implementation of binary logistic regression, built to classify *Iris-setosa* against the other two Iris species using nothing but Python's standard library.

## 1. Purpose

This project exists to prove out logistic regression as a classification algorithm by building it manually — sigmoid, cost function, gradient, and gradient descent — instead of calling a machine learning library. It was written by Carter Ward for CS 430-1 ("Problem 2" of the course), so the goal was both academic (satisfy an assigned classification problem) and practical: understand what a library like scikit-learn's `LogisticRegression` is actually doing under the hood before ever relying on it as a black box.

The project also doubles as a small, self-contained reference for anyone who wants to see a full logistic regression pipeline — data loading, preprocessing, training, and evaluation — in under 130 lines of plain Python, with no dependencies to install.

## 2. Problem and approach

The problem was assigned as part of a coursework set: implement binary logistic regression to classify a real dataset. The classic UCI **Iris dataset** (`iris.data`, 150 samples, 4 numeric features, 3 species) was chosen as the data source, and the classification task was converted from a 3-class problem into a binary one: **Iris-setosa vs. everything else** (Iris-versicolor and Iris-virginica collapsed into a single "other" class).

The approach taken was to implement the entire logistic regression pipeline by hand rather than use an existing ML library:

- Parse `iris.data` and re-label each row: `1` if the class is `Iris-setosa`, `0` otherwise.
- Add an intercept (bias) term to every feature vector.
- Split the data into an 80% training set and a 20% validation set using a fixed random seed (`random.seed(430)`) so results are reproducible.
- Train with **batch gradient descent** on the manually-derived logistic cost function and its gradient.
- Evaluate the trained model on the held-out validation split using accuracy, precision, and a confusion matrix.

This from-scratch approach was chosen deliberately over `scikit-learn` or a similar library specifically so the underlying math (log-loss, the sigmoid, and the gradient update rule) would have to be implemented and understood directly, rather than treated as a black box.

## 3. Structure and methodologies

**Language / runtime:** Python 3 (developed and tested on 3.11).

**Dependencies:** none beyond the Python standard library — only `math` (for `exp` and `log`) and `random` (for shuffling and reproducible splitting) are imported. No NumPy, pandas, or scikit-learn are used anywhere in the project, which keeps the math fully visible and auditable in the source.

**Data structures used:**
- Feature vectors and the design matrix are represented as plain Python `list`s of `list`s (`X`), rather than NumPy arrays — every dot product, sum, and elementwise operation is written out with list comprehensions and generator expressions.
- Labels (`y`) and the parameter vector (`theta`) are flat lists.
- The confusion matrix is returned as a plain 4-tuple (`tp, tn, fp, fn`) rather than a matrix object.

**Key functions / methodologies (in `WardCS430Problem02Program.py`):**
- `sigmoid(z)` — the logistic activation function, applied elementwise if given a list.
- `load_iris_data(file_path)` — reads `iris.data`, splits each CSV line, and binarizes the class label.
- `add_intercept(X)` — prepends a constant `1.0` column to the feature matrix.
- `dot(a, b)` — a manual dot product used everywhere theta is applied to a feature row.
- `compute_cost(theta, X, y)` — the binary cross-entropy (log-loss) cost function.
- `compute_gradient(theta, X, y)` — the gradient of that cost function with respect to theta.
- `gradient_descent(X, y, theta, alpha, num_iters)` — the optimization loop; run with a learning rate `alpha = 0.1` for `1000` iterations.
- `predict`, `accuracy`, `precision`, `confusion_matrix` — evaluation utilities applied to the validation split.
- `train_test_split(X, y, test_ratio=0.2)` — a hand-rolled 80/20 shuffle-and-split, seeded for reproducibility.

**Data files:**
- `iris.data` — the raw 150-row Iris dataset (sepal length/width, petal length/width, species).
- `iris.names` — the original UCI documentation for the dataset, including attribute descriptions and summary statistics (notably that Setosa is linearly separable from the other two classes, which is exactly why it was chosen as the positive class for a binary logistic model).

## 4. Process

The commit and file history tells a fairly linear, assignment-driven story:

1. **Start with the dataset.** The canonical UCI Iris files (`iris.data` and `iris.names`) were pulled in first, giving a known, well-documented, small dataset to work against — no time spent on data collection or cleaning, so the focus could stay on the algorithm itself.
2. **Reduce the problem to binary classification.** Since logistic regression in its basic form is a binary classifier, the first real design decision was collapsing the 3-class Iris problem down to "Setosa vs. not-Setosa." This was a deliberate simplification informed by `iris.names` itself, which notes that Setosa is linearly separable from the other two species — making it a clean, well-behaved target for a first from-scratch implementation.
3. **Build the math bottom-up.** The program was assembled in the same order the math is normally taught: sigmoid first, then the cost function, then its gradient, then the optimizer that ties them together (`gradient_descent`). Keeping each piece as its own small function made it possible to reason about (and debug) the gradient descent update in isolation from the data-loading and evaluation code.
4. **Add reproducible train/validation splitting.** An 80/20 split with a fixed seed (`random.seed(430)`) was introduced so that results could be reported consistently and re-checked, rather than varying on every run — an important detail for a graded assignment where the reported numbers need to be reproducible.
5. **Layer on evaluation.** Once training worked, accuracy, precision, and a confusion matrix were added as separate, single-purpose functions so the model's behavior could be inspected from more than one angle instead of trusting a single summary number.
6. **Document the run.** A `README.txt` was written documenting the purpose, files, requirements, exact run command, and a sample of the program's output, so the project could be understood and re-run without reading the source first.

Throughout, the "no external libraries" constraint shaped almost every implementation choice — vectorized operations that would be a one-liner in NumPy (dot products, elementwise sigmoid, gradient accumulation) had to be written as explicit loops and comprehensions instead.

## 5. Outcome

Running `python WardCS430Problem02Program.py` trains the model on the 80% training split and evaluates it on the held-out 20% validation split. Per the project's own documented run, the model produced:

- **Confusion matrix:** TP = 10, TN = 20, FP = 1, FN = 2
- **Accuracy:** 0.9000
- **Precision:** 0.9091

on the validation set, using a fixed seed of `430` for reproducibility.

Beyond the specific numbers, finishing this project demonstrates a working, end-to-end understanding of logistic regression: deriving and implementing the sigmoid function, the log-loss cost function, its gradient, and a batch gradient descent optimizer, all without leaning on a machine learning library to hide the details. It shows the ability to take a raw, real-world dataset, reduce a multi-class problem to a well-posed binary one, and build a complete, reproducible train/evaluate pipeline (including a controlled random split and multiple evaluation metrics) around it.

The main things learned in the process were:

- How the sigmoid, cost, and gradient equations for logistic regression translate directly into code, line by line, without any library abstraction in the way.
- Why reproducibility matters in ML experiments — and how a single fixed random seed makes a train/validation split (and therefore the reported metrics) trustworthy and repeatable.
- How to evaluate a classifier honestly using more than one metric (accuracy, precision, and a full confusion matrix) instead of a single headline number.
- Practical, from-scratch numerical Python: writing dot products, elementwise math, and gradient accumulation with plain lists and loops instead of NumPy, which builds a much closer intuition for what higher-level ML libraries are doing internally.
