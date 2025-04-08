# Machine Learning - DAT 412
This repository contains assignments completed as part of my DAT 412 Machine Learning course

---

## Assignments Overview

###  A1 – Linear Regression with Gradient Descent
- Implements univariate linear regression to model the relationship between study hours and test scores.
- Uses gradient descent to optimize weights and biases.
- Includes predictions and visualizations.
- **Key takeaway**: Understanding cost function minimization and model interpretation.

---

### A2 – Multivariate Linear Regression & Vectorization
- Predicts housing prices based on size, number of bedrooms, and age.
- Introduces feature scaling (Z-score normalization) to speed up convergence.
- Compares non-vectorized and vectorized implementations of gradient descent.
- **Key takeaway**: Vectorization improves performance and stability in large datasets.

---

### A3 – Logistic Regression for Binary Classification
- Predicts the likelihood of customer purchase using time on website and number of pages visited.
- Implements logistic regression from scratch, including sigmoid function, cost computation, gradient descent, and regularization.
- Evaluates performance using accuracy.
- **Key takeaway**: Logistic regression is ideal for binary outcomes, and regularization prevents overfitting.

---

### A4 – Neural Network for Handwritten Digit Classification
- Classifies grayscale 20x20 images of handwritten digits (0 or 1).
- Trains a feedforward neural network using TensorFlow/Keras.
- Tests different architectures (25 vs. 100 neurons, ReLU vs. tanh).
- Evaluates performance using accuracy, confusion matrix, and precision/recall.
- **Key takeaway**: Simple neural nets can effectively classify binary image data but require tuning for generalization.

---

### A5 – Deep Learning Regression Model
- Uses a deep neural network to predict prices from synthetic real estate features.
- Includes model training, validation, and evaluation using MAE (mean absolute error).
- Visualizes loss and MAE trends over epochs, and compares predicted vs. actual prices.
- **Key takeaway**: Neural networks can model nonlinear relationships in regression problems effectively with proper architecture and tuning.

---

### A6 – Neural Networks for Predicting Housing Prices
- Builds a deep neural network to predict housing prices based on size (sqft), location score, and number of bedrooms.
- Normalizes features using StandardScaler and splits data into training and testing sets (80/20).
- Trains a two-layer neural network (64 and 32 neurons with ReLU activation) using the Adam optimizer and MSE loss.
- Evaluates performance with MAE and visualizes both training vs. validation loss and true vs. predicted prices.
- **Key takeaway**: Neural networks can generalize well for price prediction tasks and benefit from feature scaling and model tuning.

---

## Technologies Used
- Python
- NumPy
- Matplotlib
- TensorFlow/Keras
- Jupyter Notebook

---

