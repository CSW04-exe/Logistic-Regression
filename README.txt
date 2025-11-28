========================================================
README - Logistic Regression on the Iris Dataset
========================================================

Author : Carter Ward  
Class  : CS 430-1  
Date   : 10/25/2025  

--------------------------------------------------------
Overview
--------------------------------------------------------
This project implements binary logistic regression from scratch
using the classic Iris dataset.

The main program (WardCS430Problem02Program.py):
- Loads data from iris.data
- Converts class labels to binary: Iris-setosa vs. all others
- Splits the data randomly into 80% training and 20% validation
- Implements the sigmoid, cost function, and gradient manually
- Uses batch gradient descent to optimize parameters (θ)
- Predicts labels for the validation set
- Reports the confusion matrix, accuracy, and precision

A diagram of the sigmoid curve is optionally included to satisfy
Problem 1 from the written portion.

--------------------------------------------------------
Files
--------------------------------------------------------
iris.data                    -> dataset (150 rows, 4 features + class)
WardCS430Problem02Program.py -> main program with full implementation
sigmoid_plot.png             -> (optional) plot of sigmoid function
README.txt                   -> this file

--------------------------------------------------------
Requirements
--------------------------------------------------------
- Python 3.x (tested on 3.11)
- No external libraries required
- Only built-in modules (math, random) are used

--------------------------------------------------------
How to Run
--------------------------------------------------------
1. Make sure iris.data is in the same folder as WardCS430Problem02Program.py
2. Open a terminal in that folder (or run from VS Code)
3. Execute the program:
    python WardCS430Problem02Program.py

--------------------------------------------------------
Sample Output
--------------------------------------------------------
Optimal theta: [values...]

Confusion Matrix: TP=10, TN=20, FP=1, FN=2  
Accuracy: 0.9000  
Precision: 0.9091

--------------------------------------------------------
Notes
--------------------------------------------------------
To ensure reproducibility, the training/validation split uses a
fixed seed (random.seed(430)).


