# Batch vs Stochastic Gradient Descent

## Overview
Explored the effect of batch size on neural network training using a binary classification task. Compared training behaviour with different batch sizes using TensorFlow and Keras.

## Dataset
Used the Social Network Ads dataset to predict whether a user purchases a product based on:

- Age
- Estimated Salary

**Target:** `Purchased` (binary classification)

## Data Preprocessing
- Selected the relevant input features and target variable.
- Applied `StandardScaler` to standardize the input features.
- Used an 80/20 training-validation split through Keras' `validation_split`.

## Model Architecture
- Input features: 2
- Hidden layers: 2, with 10 neurons each
- Output layer: 1 neuron with sigmoid activation
- Loss function: Binary cross-entropy
- Optimizer: Keras default optimizer (RMSprop)

## Experiments

### Experiment 1: Stochastic Gradient Descent Style
- Batch size: 1
- Epochs: 500

With a batch size of 1, the model processes one training example per batch, resulting in frequent parameter updates.

### Experiment 2: Mini-Batch Gradient Descent
- Batch size: 150
- Epochs: 10

With a batch size of 150, the model processes groups of training examples before updating parameters.

### Experiment 3: Full-Batch Gradient Descent
- Batch size: 250
- Epochs: 10

Compared training behaviour with a batch size large enough to process the entire training split in one batch.

## Visualisation
Plotted the training loss to observe how batch size affects the model's learning behaviour.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

## Learning Outcomes
- Understood the relationship between batch size and parameter-update frequency.
- Compared stochastic, mini-batch, and full-batch training behaviour.
- Observed training loss using Keras history.


