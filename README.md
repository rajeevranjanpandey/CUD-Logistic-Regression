# Logistic Regression from Scratch with Gradient Descent

Course project for MAI 600, Mathematics and Statistics for AI.

**Group members:** Rajeev Ranjan Pandey  and Hussein Aboueldahab 

A binary logistic regression classifier written from first principles in NumPy and trained with batch gradient descent. The model predicts whether a flight arrives 15 or more minutes late, using two features: departure delay in minutes and scheduled departure hour.

No scikit-learn, TensorFlow or PyTorch is used. The prediction, the cost, the gradients, the training loop, the train/test split, the feature scaling and the evaluation metrics are all hand-written.

---

## Thought Process and Implementation Approach

### Why logistic regression?

The assignment asks for a binary classifier trained by gradient descent, implemented without machine learning libraries. Logistic regression is the natural starting point: it has a closed-form probabilistic output, a convex loss function, and an analytically tractable gradient. That means we can derive every update rule on paper before writing a single line of code, and then verify the derivation numerically. It also means the training loop is transparent -- there are no hidden layers, no black-box optimizers, just weights moving down a smooth surface.

### Choosing the dataset and features

We wanted a dataset where the relationship between the features and the target is physically meaningful, not just statistically convenient. Flight departure delays have a direct causal link to arrival delays: a flight that leaves late almost always arrives late. Scheduled departure hour matters too, because early-morning flights tend to have fewer downstream delays than afternoon or evening flights that have already absorbed schedule slippage across the day.

We sampled 1,000 rows from the nycflights13 dataset (all flights departing New York City's three airports in 2013) using a fixed random seed so the sample is reproducible. The target variable is `late_arrival`, defined as an arrival delay of 15 minutes or more. That threshold is a standard industry definition of a late arrival, which makes the binary classification boundary meaningful rather than arbitrary. Of the 1,000 flights in the sample, 243 arrive late and 757 arrive on time, giving a moderate class imbalance that the model has to handle honestly -- we do not resample or re-weight the classes.

### Data preparation

Before any model training, we split the data into 800 training flights and 200 test flights using a hand-written stratified split. Stratification ensures both splits preserve the 24.3% late-arrival rate, so the test set is representative and evaluation metrics are reliable. The split uses a fixed seed of 42 throughout.

We then standardize the features using z-score normalization: subtract the training mean and divide by the training standard deviation. The key discipline here is that the normalization statistics come from the training set only. Applying them to the test set without recomputing them prevents any information from the test set leaking into the model, which would make our accuracy figures optimistic and unreliable.

Departure delay in minutes ranges from large negative values (early departures) to several hundred minutes late. Scheduled departure hour runs from 5 to 23. Without standardization, gradient descent would behave poorly because the two features live on completely different scales, causing the loss surface to be elongated along one axis and making learning rate selection difficult. After standardization both features have zero mean and unit variance, and the gradient steps are balanced.

### Mathematical foundations

The model computes a linear combination of the two standardized features:

```
z = w1 * dep_delay_min_std + w2 * sched_dep_hour_std + b
```

This is passed through the sigmoid function to produce a probability:

```
sigmoid(z) = 1 / (1 + exp(-z))
```

The sigmoid maps any real number to the interval (0, 1), which we interpret as the probability that the flight arrives late. We classify a flight as late when that probability exceeds 0.5.

The model is trained by minimizing binary cross-entropy, also called log loss:

```
J(w, b) = -(1/m) * sum[ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]
```

This loss penalizes confident wrong predictions heavily (because log approaches negative infinity near zero) and rewards confident correct predictions. It is convex in the weights, which guarantees that gradient descent converges to the global minimum rather than a local one.

The gradients with respect to the weights and bias are derived by applying the chain rule through the loss, sigmoid, and linear layers. The result is clean:

```
dJ/dw = (1/m) * X^T * (y_hat - y)
dJ/db = (1/m) * sum(y_hat - y)
```

The gradient is simply the residual (predicted minus actual) averaged over the training set, weighted by the input features for the weights. This is the same form as the gradient for linear regression, which makes intuitive sense: the sigmoid is just wrapping the linear combination.

Each gradient descent step updates the parameters:

```
w = w - alpha * dJ/dw
b = b - alpha * dJ/db
```

where `alpha` is the learning rate.

### Gradient verification

Before running the full training loop, we verify the analytical gradients numerically using central differences. For each parameter, we perturb it by a small epsilon in both directions and estimate the gradient from the change in cost:

```
numerical_gradient ~ (J(theta + eps) - J(theta - eps)) / (2 * eps)
```

The difference between the analytical and numerical gradients should be on the order of machine epsilon for double-precision arithmetic. In practice we observe differences around 3e-11, which confirms the derivation is correct before we spend any time watching the training loop.

### Training and hyperparameter comparison

We train the model with three learning rates: 0.1, 0.01, and 0.001, each for 1,000 iterations of batch gradient descent. All 800 training examples are used in every gradient step -- this is full-batch gradient descent, not stochastic or mini-batch. We also run a reference training of 50,000 iterations at learning rate 0.1 to show what the cost looks like after near-convergence.

Learning rate 0.1 converges well within 1,000 iterations, reaching a final cost of approximately 0.305. Learning rate 0.01 makes slower progress and reaches around 0.370 after 1,000 iterations. Learning rate 0.001 has barely moved by iteration 1,000, with a cost still above 0.58. This illustrates the fundamental tradeoff: a larger learning rate covers more ground per iteration but risks overshooting; a smaller rate is more cautious but needs far more iterations to converge.

We record decision boundary snapshots at iterations 1, 10, 100, and 1,000 to visualize how the boundary evolves from its random initialization toward the learned position. Early boundaries are nearly vertical or horizontal (the random weights are small and roughly equal); by iteration 1,000 the boundary has rotated to reflect the much stronger predictive power of departure delay compared to departure hour.

### Evaluation

On the held-out 200 test flights, the model trained with learning rate 0.1 achieves:

- Accuracy: 92.0%
- Precision: 92.3%
- Recall: 73.5%
- F1-score: 81.8%

The confusion matrix shows 148 true negatives, 36 true positives, 3 false positives, and 13 false negatives. The model is conservative: it misses about a quarter of the actual late arrivals (false negatives) but rarely flags an on-time flight as late (very few false positives). This asymmetry is partly a consequence of the class imbalance -- with 75.7% of flights on time, a model that defaults to predicting "on time" would achieve 75.7% accuracy while identifying no late flights at all. Our model's recall of 73.5% on the minority class is therefore a meaningful result, not a trivially high number.

---

## Files

| File | Purpose |
| --- | --- |
| `logistic_regression.py` | Model class, training, evaluation and all plots |
| `prepare_dataset.py` | Builds the 1,000-flight sample from the nycflights13 dataset |
| `flights_two_features.csv` | The dataset the model reads |
| `results.json` | Every number reported, written by the main script |
| `figures/` | The six figures, created when the main script runs |

## How to run

```bash
pip install numpy pandas matplotlib
python logistic_regression.py
```

The CSV is included, so `prepare_dataset.py` only needs to be run to rebuild it (`pip install nycflights13` first).

## Results on the test set

200 unseen flights, learning rate 0.1, 1000 iterations.

| Accuracy | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| 0.9200 | 0.9231 | 0.7347 | 0.8182 |

Learning rates of 0.01 and 0.001 follow the same path more slowly and have not converged after 1,000 iterations.

## Data source

nycflights13, all flights that departed New York City's three airports in 2013: https://github.com/tidyverse/nycflights13
