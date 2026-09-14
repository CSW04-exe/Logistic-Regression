# Logistic Regression — Iris Classification

**Type:** Individual project
**Contributor:** Carter Ward
**Course:** CS 430-1 (Machine Learning) — Problem 2
**Completed:** 10/25/2025

## Purpose

This is my Problem 2 submission for CS 430. The assignment asked me to implement logistic regression myself rather than call a library, so `WardCS430Problem02Program.py` builds the sigmoid function, cost function, gradient, and gradient descent optimizer entirely from scratch in plain Python — no `sklearn`.

## Problem and Approach

The task is to classify flowers in the Iris dataset (`iris.data`, 150 samples, 4 features, 3 species). Since basic logistic regression is a binary classifier, I framed this as a binary one-vs-rest problem: Iris-setosa (label 1) vs. the other two species combined (label 0). Setosa is linearly separable from the rest, so a linear decision boundary is appropriate.

## Structure and Methodologies

- `X`: feature matrix as a list of rows (4 floats each); `add_intercept(X)` prepends a `1.0` bias term to each row.
- `theta`: weight vector (length 5); `y`: binary label vector.
- `sigmoid(z) = 1 / (1 + e^-z)` applied to `theta · x` for predictions.
- `compute_cost`: binary cross-entropy cost function.
- `compute_gradient` / `gradient_descent`: hand-written batch gradient descent, `alpha = 0.1`, `1000` iterations.
- Stdlib only — `math` (exp/log) and `random` (shuffling, splitting); no numpy/pandas/sklearn.

## Process

1. Load `iris.data` and map species to binary labels.
2. Add the intercept term to each feature row.
3. Shuffle (seed `430`) and split 80/20 into train/validation sets.
4. Initialize `theta` to zeros; set `alpha = 0.1`, `1000` iterations.
5. Train via full-batch gradient descent.
6. Predict on the validation set (sigmoid + 0.5 threshold).
7. Evaluate: confusion matrix, accuracy, precision.

## Outcome

With the fixed seed, the model learned `theta = [0.3535, 0.5415, 1.7825, -2.8265, -1.3190]` and produced a confusion matrix of TP=13, TN=17, FP=0, FN=0 on the 30-example validation split — 100% accuracy and 100% precision, with zero false positives or negatives. This matches expectations, since Setosa is linearly separable from the other two species. Implementing the sigmoid, cross-entropy cost, and gradient by hand made it clear why gradient descent works, and building the evaluation pipeline from scratch gave me a much better feel for how model evaluation works under the hood.

## How to Run

```
python3 WardCS430Problem02Program.py
```
`iris.data` must be in the same folder. No external dependencies required.
