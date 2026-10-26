# MNIST Classification using ANN

A hands-on Deep Learning project that uses an Artificial Neural Network (ANN) to classify handwritten digits from the MNIST dataset.

## What This Notebook Covers
- Loading and exploring the MNIST dataset using Keras.
- Visualizing handwritten digit images.
- Normalizing pixel values to the range 0–1.
- Building a neural network using Keras Sequential API.
- Training the model and tracking validation performance.
- Predicting digit classes using the model's output probabilities.
- Evaluating performance using accuracy.

## Model Architecture
- Flatten layer to convert 28 × 28 images into a one-dimensional vector.
- Hidden layer with 128 neurons and ReLU activation.
- Hidden layer with 32 neurons and ReLU activation.
- Output layer with 10 neurons and Softmax activation.

## Model Configuration
- Optimizer: Adam
- Loss function: Sparse categorical cross-entropy
- Training epochs: 25

## Results
- Test accuracy: **97.55%**

## Technologies Used
- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Scikit-learn

## Learning Objective
To understand how an ANN learns to classify handwritten digits into ten classes, from 0 to 9.

