# Early Stopping in Neural Networks

## Overview

This experiment explores **Early Stopping** as a regularization technique to reduce overfitting in neural networks. Using the `make_circles` dataset, I trained a neural network and compared its training and validation loss to understand when training should stop.

## Dataset

- **Dataset:** Synthetic circles dataset generated using `make_circles`
- **Samples:** 100
- **Noise:** 0.1
- **Random state:** 1
- **Task:** Binary classification

## Model Architecture

- Input: 2 features
- Hidden layer: 256 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation
- Loss function: Binary cross-entropy
- Optimizer: Adam

## Experiment 1: Training Without Early Stopping

The model was trained for up to 3,500 epochs without an early-stopping callback.

The training and validation loss curves helped reveal how the model's performance changed across epochs. While training loss continued to decrease, validation loss began to increase, indicating **overfitting**.

## Experiment 2: Training With Early Stopping

The same architecture was trained with the `EarlyStopping` callback.

```python
callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=20,
    verbose=1,
    mode="auto",
    restore_best_weights=False
)
```

The callback monitored validation loss and stopped training when the monitored metric failed to improve sufficiently for the specified patience period. Training stopped before reaching the maximum of 3,500 epochs.

## Key Observations

- **Overfitting:** Training loss decreased while validation loss increased, suggesting that the model was fitting the training data without improving generalization.
- **Early stopping:** Training terminated before the maximum epoch limit when validation loss stopped improving.
- **Decision regions:** Visualizing the model's predictions helped inspect how the network separated the two classes.
- **Training efficiency:** Early stopping avoided continuing training unnecessarily after validation performance stopped improving.

## Conclusion

This experiment demonstrated how validation loss can be used to identify overfitting and control the duration of neural network training. Early Stopping provides a practical way to limit unnecessary training and potentially improve generalization.

**Note:** `restore_best_weights=False` was used in this experiment. Consequently, the model retained the weights from the stopping epoch rather than automatically restoring the weights from the epoch with the lowest validation loss.

