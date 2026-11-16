# Logistic Regression from Scratch with Gradient Descent

Course project for MAI 600, Mathematics and Statistics for AI.

**Group members:** Rajeev Ranjan Pandey (20250007918) and Hussein Aboueldahab (20250008083)

A binary logistic regression classifier written from first principles in NumPy and trained with batch gradient descent. The model predicts whether a flight arrives 15 or more minutes late, using two features: departure delay in minutes and scheduled departure hour.

No scikit-learn, TensorFlow or PyTorch is used. The prediction, the cost, the gradients, the training loop, the train/test split, the feature scaling and the evaluation metrics are all hand-written.

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

Learning rates of 0.01 and 0.001 follow the same path more slowly and have not converged after 1000 iterations.

## Data source

nycflights13, all flights that departed New York City's three airports in 2013: https://github.com/tidyverse/nycflights13
