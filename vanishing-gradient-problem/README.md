# Vanishing Gradient Problem

## Overview
Explored the vanishing gradient problem by comparing neural networks of different depths on a binary classification task using the `make_moons` dataset.

The experiment investigates how network depth and activation functions can affect weight updates and the learning process.

## Dataset
Generated a synthetic two-class dataset using Scikit-learn's `make_moons()`.

- Samples: 250
- Noise: 0.05
- Random state: 42
- Input features: 2
- Target classes: 0 and 1

## Model Experiments

### Experiment 1: Shallow Network with Sigmoid
Built a neural network with three hidden layers containing 10 neurons each, followed by a sigmoid output layer.

- Hidden activation: Sigmoid
- Output activation: Sigmoid
- Loss: Binary cross-entropy
- Optimizer: Adam
- Epochs: 100

Recorded the initial and final weights to examine how parameters changed during training.

### Experiment 2: Deeper Network with ReLU
Built a deeper neural network containing eleven hidden layers with 10 neurons each, followed by a sigmoid output layer.

- Hidden activation: ReLU
- Output activation: Sigmoid
- Loss: Binary cross-entropy
- Optimizer: Adam
- Epochs: 100

Compared initial and final weights to investigate how network depth and activation functions influence parameter updates.

## Observations

- The shallow sigmoid network showed changes in its weights during training.
- In the deeper network, the loss appeared to plateau around 1.000 during training.
- This raised the question of whether gradient propagation and parameter updates were becoming ineffective as network depth increased.
- The experiment motivated further investigation into vanishing gradients and the effect of activation functions on deep neural networks.

## Important Note
A loss value plateauing around 1.000 does not independently establish that vanishing gradients occurred. Gradient values across layers, weight updates, and training behaviour must be examined to confirm the cause.

The weight comparison in this experiment focuses only on the first layer. Additional measurements across multiple layers would provide stronger evidence.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- TensorFlow / Keras
- Scikit-learn

## Learning Outcomes
- Explored neural network depth and activation functions.
- Compared initial and trained weights.
- Investigated loss plateaus during training.
- Studied the relationship between gradient propagation and deep network training.

## Next Steps
- Plot training loss for both architectures.
- Measure gradient magnitudes across layers.
- Compare sigmoid and ReLU networks under controlled conditions.
- Evaluate model accuracy on the held-out test set.

