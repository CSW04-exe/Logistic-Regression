# Logistic Regression on the Iris Dataset

**Author:** Carter Ward
**Class:** CS 430-1 (UAH)
**Date:** 10/25/2025

## Purpose

This is my Problem 2 submission for CS 430, a machine-learning course. The
assignment asked me to implement logistic regression myself rather than call
a library, so `WardCS430Problem02Program.py` builds the sigmoid function,
cost function, gradient, and gradient descent optimizer entirely from
scratch in plain Python. The point of the exercise was to make sure I
actually understood the mechanics of logistic regression — not just how to
call `sklearn.linear_model.LogisticRegression`.

## Problem and Approach

The task is to classify flowers in the classic Iris dataset (`iris.data`,
150 samples, 4 numeric features: sepal length, sepal width, petal length,
petal width, across 3 species — Setosa, Versicolour, Virginica). Since
logistic regression in its basic form is a binary classifier, I framed this
as a **binary, one-vs-rest problem**: Iris-setosa (label 1) versus the other
two species combined (label 0). Setosa is a good choice for this because it
is linearly separable from the other two classes (the other two are not
linearly separable from each other, per `iris.names`), so a linear decision
boundary is actually appropriate here.

My approach was fully manual:
- No `numpy`, `pandas`, or `scikit-learn` — only Python's built-in `math`
  and `random` modules.
- Vectors and matrices are represented as plain Python lists and lists of
  lists, with helper functions (`dot`, `sigmoid`) standing in for what a
  library like NumPy would normally provide.
- Gradient descent is a hand-written loop that updates the parameter vector
  θ (theta) iteratively.

## Structure and Methodologies

**Data structures**
- `X`: the feature matrix, a list of rows, each row a list of 4 floats
  (sepal length, sepal width, petal length, petal width).
- After `add_intercept(X)`, each row gets a leading `1.0` so the model can
  learn a bias term — `X` becomes a 150x5 matrix conceptually.
- `theta`: the weight vector (length 5: bias + 4 feature weights).
- `y`: the binary label vector (1 = Setosa, 0 = other).

**Math**
- **Sigmoid activation:** `sigmoid(z) = 1 / (1 + e^-z)`, applied to the dot
  product of `theta` and a feature row to squash the linear score into a
  probability between 0 and 1.
- **Cost function:** the standard logistic regression (binary
  cross-entropy) cost,
  `J(theta) = -(1/m) * sum( y*log(h) + (1-y)*log(1-h) )`,
  implemented directly in `compute_cost`.
- **Gradient:** `compute_gradient` computes
  `(1/m) * sum( (sigmoid(theta·x) - y) * x )` for each parameter, which is
  the analytical gradient of the cross-entropy cost with respect to theta.
- **Batch gradient descent:** `gradient_descent` repeatedly computes the
  full-batch gradient over all training examples and updates
  `theta = theta - alpha * grad`, for a fixed learning rate `alpha = 0.1`
  and `num_iters = 1000`.

**Dependencies:** none beyond the Python standard library (`math` for
`exp`/`log`, `random` for shuffling and the train/validation split). No
plotting or numerical libraries are imported by the program itself.

## Process

Running `main()` walks through these steps in order:

1. **Load the data.** `load_iris_data("iris.data")` reads each line of the
   dataset, splits on commas, converts the first four columns to floats,
   and maps the class column to a binary label (`1` for `Iris-setosa`, `0`
   for anything else).
2. **Add the intercept term.** `add_intercept(X)` prepends a `1.0` to every
   feature row so the learned model includes a bias weight.
3. **Split into train/validation sets.** `train_test_split` shuffles the
   150 examples (using a fixed `random.seed(430)` for reproducibility) and
   splits them 80/20 into training and validation sets.
4. **Initialize parameters.** `theta` starts as a zero vector of length 5;
   the learning rate is `alpha = 0.1` and training runs for `1000`
   iterations.
5. **Train via batch gradient descent.** On each iteration, the program
   computes the gradient of the cost function over the *entire* training
   set and takes a step in the direction that reduces the cost, gradually
   converging on the parameters that best separate Setosa from the other
   species.
6. **Predict on the validation set.** `predict` applies the sigmoid to each
   validation row's dot product with the learned theta and thresholds at
   0.5 to produce a class label.
7. **Evaluate.** The program computes a confusion matrix (TP/TN/FP/FN),
   accuracy, and precision by comparing predicted labels to the true
   validation labels, then prints the learned theta and all metrics.

## Outcome

Running the program (`python3 WardCS430Problem02Program.py`) with the fixed
seed produces:

```
Optimal theta: [0.3535, 0.5415, 1.7825, -2.8265, -1.3190]
Confusion Matrix: TP=13, TN=17, FP=0, FN=0
Accuracy: 1.0000
Precision: 1.0000
```

The model reaches **100% accuracy and 100% precision** on the 30-example
validation split, with zero false positives and zero false negatives. This
matches expectations from the dataset's documentation (`iris.names`), which
notes that Setosa is linearly separable from the other two species — so a
simple linear decision boundary, learned by plain gradient descent, is
enough to separate it perfectly.

Working through this assignment forced me to actually understand the math
behind logistic regression rather than treat it as a black box: deriving
and implementing the sigmoid, the cross-entropy cost function, and its
gradient by hand made it clear *why* gradient descent works the way it
does, and building the train/validation split and evaluation metrics
(confusion matrix, accuracy, precision) from scratch gave me a much better
feel for how model evaluation actually works under the hood. It's a small,
clean dataset, so the perfect score isn't surprising, but getting there
with a from-scratch implementation was a good gut-check that my
understanding of the underlying math and optimization procedure was
correct.

## How to Run

1. Make sure `iris.data` is in the same folder as
   `WardCS430Problem02Program.py`.
2. Run:
   ```
   python3 WardCS430Problem02Program.py
   ```

No external dependencies are required — only the Python standard library.
