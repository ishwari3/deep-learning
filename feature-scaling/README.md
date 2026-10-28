# Feature Scaling in Neural Networks

## Overview

Feature scaling is a preprocessing technique that brings numerical features to comparable scales. It is particularly useful when training neural networks, where features with significantly different ranges can affect the optimization process.

In this exercise, I explored how **Standardization affects neural-network training and validation accuracy** using the Social Network Ads dataset.

## Why Does Feature Scaling Matter?

Imagine training a model using two features:

- **Age:** ranges approximately from 18 to 60.
- **Salary:** ranges from thousands to hundreds of thousands.

Although both features can contain useful information, their numerical scales are very different.

When features have significantly different scales, optimization can become less efficient. In gradient-based learning, this may lead to uneven updates, oscillations during optimization, and slower convergence.

For example, if we visualize the loss surface, unscaled features can produce an elongated, poorly conditioned surface. Gradient descent may zigzag while approaching a minimum instead of taking a more direct path.

Feature scaling helps address this problem by making the numerical scales of the input features more comparable.

**Important:** A larger numerical range does not automatically mean that a neural network ignores another feature. Scaling helps optimization; it does not determine a feature's importance by itself.

## Types of Feature Scaling

### 1. Normalization (Min-Max Scaling)

Min-max scaling transforms values into a specified range, commonly \([0,1]\).

Formula:

\[
X_{\text{scaled}}=\frac{X-X_{\min}}{X_{\max}-X_{\min}}
\]

It is useful when a bounded range is desired. For example, CGPA on a 0–10 scale can be transformed to a 0–1 range.

However, normalization can be sensitive to outliers because the minimum and maximum values determine the transformation.

### 2. Standardization (Z-score Scaling)

Standardization transforms a feature so that it has a mean of approximately 0 and a standard deviation of approximately 1.

Formula:

\[
X_{\text{scaled}}=\frac{X-\mu}{\sigma}
\]

Where:
- \(X\) is the original value.
- \(\mu\) is the feature's mean.
- \(\sigma\) is the feature's standard deviation.

Standardization is useful when features have different ranges and we want to scale them relative to their distributions. It does not constrain values to a fixed range.

**Choosing between them:** Neither method is universally better. The choice depends on the data, the model, and the problem being solved.

## Implementation

### Dataset and Model

I used the Social Network Ads dataset, with **Age** and **Estimated Salary** as input features and the purchase outcome as the target.

The neural network consists of:

- Input: 2 features
- Hidden layer: 128 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation
- Optimizer: Adam
- Loss function: Binary cross-entropy
- Training duration: 100 epochs

### Experiment 1: Without Feature Scaling

I first trained the neural network using the original input features.

The validation-accuracy graph fluctuated considerably, with values moving roughly between **0.35 and 0.60** in this run.

These fluctuations indicate that validation performance was unstable across epochs. The model was learning from features with substantially different numerical ranges, which can make optimization less well-conditioned.

### Experiment 2: With Standardization

Next, I applied `StandardScaler` to the training features and used the same fitted scaler to transform the test features.

The validation-accuracy graph showed a clear improvement in this run: accuracy rose from approximately **0.79 initially to around 0.92–0.94**, before stabilizing.

This suggests that scaling helped the optimizer learn more effectively on this dataset.

The exact values can vary with the data split, initialization, and training run. Validation accuracy alone also does not establish that the model will generalize well to unseen data.

## Implementation with Scikit-learn

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit only on the training data
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same transformation to test data
X_test_scaled = scaler.transform(X_test)
```

The scaler learns the mean and standard deviation from the training data. Applying that same transformation to the test data avoids **data leakage**.

I then rebuilt and trained the neural network using `X_train_scaled` and evaluated it using `X_test_scaled`.

## Results and Observations

| Observation | Without scaling | With standardization |
|---|---|---|
| Input features | Original numerical ranges | Comparable standardized scales |
| Validation accuracy in this run | Fluctuated around 0.35–0.60 | Rose to approximately 0.92–0.94 |
| Training behaviour | Less stable validation performance | Improved validation performance |
| Overall takeaway | Optimization can be less efficient | Scaling can improve optimization |

These results are observations from this particular experiment, not a guarantee that standardization will always produce higher accuracy.

## Key Learnings

- Neural networks can benefit from scaling numerical input features.
- Features with very different scales can make gradient-based optimization less efficient.
- Standardization uses the training data's mean and standard deviation.
- Min-max normalization maps values into a chosen range.
- The scaler must be fitted on training data only and then reused on validation and test data.
- Comparing training and validation curves helps reveal whether a model is learning effectively and whether overfitting may be occurring.

## Conclusion

This experiment demonstrated how feature scaling can affect neural-network training. After applying standardization, the validation-accuracy curve improved substantially in my run.

It reinforced an important lesson: **preprocessing is not separate from model building; it can directly influence how effectively a model learns.**
