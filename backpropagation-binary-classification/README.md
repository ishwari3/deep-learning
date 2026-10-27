# Backpropagation for Binary Classification

## Overview
Implemented a neural network from scratch using NumPy to perform binary classification. The model predicts whether a student gets placed based on their CGPA and profile score.

## Dataset
A small sample dataset containing:

- **CGPA** — input feature
- **Profile Score** — input feature
- **Placed** — target label (`1` = placed, `0` = not placed)

## Model Architecture
- Input layer: 2 features
- Hidden layer: 2 neurons
- Output layer: 1 neuron
- Architecture: `[2, 2, 1]`

## Implementation
- Initialized weights and biases manually.
- Implemented sigmoid activation.
- Performed forward propagation across the network.
- Updated weights and biases using manually implemented backpropagation.
- Used binary cross-entropy loss to evaluate predictions.
- Trained the network over multiple epochs.

## Technologies Used
- Python
- NumPy
- Pandas
- Jupyter Notebook

## Learning Outcomes
- Applied backpropagation to a binary classification problem.
- Explored sigmoid activation and binary cross-entropy loss.
- Practiced implementing neural network training without a deep learning framework.

## Next Steps
- Correct and verify gradient calculations.
- Track and visualize training loss across epochs.
- Evaluate predictions on the training samples.
