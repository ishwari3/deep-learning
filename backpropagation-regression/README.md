# Backpropagation for Regression

## Overview
Implemented a neural network for regression using manually coded forward propagation and backpropagation to understand how weights and biases are updated during training.

## Dataset
The model uses the following input features to predict the target variable:

- **CGPA**
- **Profile Score**
- **Target:** LPA

## Model Architecture
- Input layer: 2 neurons
- Hidden layer: 2 neurons
- Output layer: 1 neuron
- Architecture: `[2, 2, 1]`

## Implementation
- Implemented forward propagation using NumPy.
- Calculated parameter updates manually using the prediction error.
- Updated weights and biases during training.
- Trained the model over multiple epochs.
- Tracked mean squared error (MSE) to monitor training loss.

## Technologies Used
- Python
- NumPy
- Pandas
- Jupyter Notebook

## Learning Outcomes
- Understood the mechanics of forward propagation.
- Explored how prediction errors influence weight and bias updates.
- Practiced implementing neural network training without relying on a deep learning framework.

## Next Steps
- Refine gradient calculations and parameter updates.
- Experiment with different learning rates and epoch counts.
- Visualize training loss across epochs.

