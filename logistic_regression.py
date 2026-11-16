"""
MAI 600 - Mathematics and Statistics for AI
Project: Logistic Regression classifier trained with Gradient Descent

Everything that matters is written from scratch with NumPy:
    - the sigmoid and the prediction (hypothesis)
    - the binary cross-entropy cost
    - the gradients and the gradient descent loop
    - the train/test split, the feature scaling and the evaluation metrics

Allowed libraries only: NumPy (maths), Pandas (loading the CSV), Matplotlib (plots).
No scikit-learn, TensorFlow or PyTorch is imported anywhere in this file.

Run with:  python logistic_regression.py
Outputs:   figures/*.png  and  results.json
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # draw straight to files, no window needed
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 0. Settings kept in one place so they are easy to find and change
# ----------------------------------------------------------------------------
DATA_FILE = "flights_two_features.csv"   # built by prepare_dataset.py
FIGURE_DIR = "figures"
SEED = 42                               # fixed seed so every run gives the same split
TEST_FRACTION = 0.20                    # 80 % training, 20 % testing
LEARNING_RATES = [0.1, 0.01, 0.001]     # the values the brief asks us to compare
MAIN_LEARNING_RATE = 0.1                # the model we report in detail
N_ITERATIONS = 1000                     # number of gradient descent updates
SNAPSHOT_ITERATIONS = [1, 10, 100, 1000]  # stages at which we draw the boundary
REFERENCE_ITERATIONS = 50000            # one long run, used only to see where the cost finally settles

FEATURES = ["dep_delay_min", "sched_dep_hour"]
FEATURE_LABELS = ["Departure delay (minutes)", "Scheduled departure time (hour of day)"]
TARGET = "late_arrival"                 # 1 = arrived 15+ minutes late (positive class), 0 = on time
CLASS_NAMES = {0: "On time (0)", 1: "Late (1)"}
X_AXIS_MAX = 120                        # plots stop here so the busy region is readable; longer delays are counted in a note
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
CLASS_MARKERS = {0: "o", 1: "^"}        # shape as well as colour, for readers who cannot rely on colour


# ----------------------------------------------------------------------------
# 1. Data preprocessing helpers (written by hand, no scikit-learn)
# ----------------------------------------------------------------------------
def load_data(path):
    """Step 1a: load the CSV and separate features X from the target y."""
    data = pd.read_csv(path)
    X = data[FEATURES].to_numpy(dtype=float)   # shape (m, 2)
    y = data[TARGET].to_numpy(dtype=float)     # shape (m,), values 0 or 1
    return X, y


def stratified_train_test_split(X, y, test_fraction, seed):
    """
    Step 1b: split into training and test sets.

    The split is stratified, which means each class is shuffled and divided
    separately. Both sets then keep the same share of late flights as the
    full dataset, so the test metrics are not distorted by an unlucky draw.
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]      # rows that belong to this class
        rng.shuffle(idx)                   # random order
        n_test = int(round(len(idx) * test_fraction))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = rng.permutation(train_idx)  # mix the classes again
    test_idx = rng.permutation(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train, X_test):
    """
    Step 1c: z-score scaling, z = (x - mean) / std.

    The mean and standard deviation come from the TRAINING set only and are
    then reused on the test set. Using test data here would leak information.
    Scaling puts both features on a similar range, which makes the cost
    surface rounder and lets one learning rate suit both weights.
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    return (X_train - mu) / sigma, (X_test - mu) / sigma, mu, sigma


# ----------------------------------------------------------------------------
# 2. The model class
# ----------------------------------------------------------------------------
class LogisticRegressionGD:
    """Binary logistic regression trained with batch gradient descent."""

    def __init__(self, learning_rate=0.1, n_iterations=1000, snapshot_iterations=None):
        # Hyperparameters (chosen by us, not learned)
        self.learning_rate = learning_rate        # alpha, the step size
        self.n_iterations = n_iterations          # how many updates to perform
        self.snapshot_iterations = set(snapshot_iterations or [])

        # Parameters (learned from data). They are created in fit() because
        # the number of weights depends on the number of features.
        self.w = None                             # weight vector, one weight per feature
        self.b = 0.0                              # bias (intercept)

        # Training diary
        self.cost_history = []                    # J after every iteration (index 0 = before training)
        self.snapshots = {}                       # {iteration: (w copy, b)} for the boundary plots

    # -- formula 1: sigmoid ---------------------------------------------------
    def sigmoid(self, z):
        """sigma(z) = 1 / (1 + e^(-z)). Maps any real number into (0, 1)."""
        z = np.clip(z, -500, 500)                 # avoids overflow in exp for extreme z
        return 1.0 / (1.0 + np.exp(-z))

    # -- formula 2: hypothesis ------------------------------------------------
    def predict_proba(self, X):
        """h(x) = sigma(w . x + b), the estimated probability that y = 1."""
        z = X @ self.w + self.b                   # linear score for every sample at once
        return self.sigmoid(z)

    # -- formula 3: cost ------------------------------------------------------
    def compute_cost(self, X, y):
        """Binary cross-entropy J(w, b) = -(1/m) * sum[ y ln(h) + (1 - y) ln(1 - h) ]."""
        m = len(y)
        h = self.predict_proba(X)
        h = np.clip(h, 1e-15, 1 - 1e-15)          # keeps ln() away from ln(0)
        return -(1.0 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    # -- formula 4: gradients -------------------------------------------------
    def compute_gradients(self, X, y):
        """
        dJ/dw_j = (1/m) * sum[ (h_i - y_i) * x_ij ]
        dJ/db   = (1/m) * sum[ (h_i - y_i) ]
        """
        m = len(y)
        error = self.predict_proba(X) - y         # (h - y) for every sample
        dw = (1.0 / m) * (X.T @ error)            # one partial derivative per weight
        db = (1.0 / m) * np.sum(error)
        return dw, db

    # -- formula 5: gradient descent ------------------------------------------
    def fit(self, X, y):
        """Learn w and b by repeating the update rule n_iterations times."""
        n_features = X.shape[1]
        self.w = np.zeros(n_features)             # start from zero: h = 0.5 for every sample
        self.b = 0.0
        self.cost_history = [self.compute_cost(X, y)]   # cost before any update, equals ln(2)
        self.snapshots = {}

        for iteration in range(1, self.n_iterations + 1):
            # 1) slope of the cost at the current position
            dw, db = self.compute_gradients(X, y)

            # 2) step downhill:  w := w - alpha * dJ/dw ,  b := b - alpha * dJ/db
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            # 3) record the cost so we can plot the learning curve
            self.cost_history.append(self.compute_cost(X, y))

            # 4) keep a copy of the parameters at the requested stages
            if iteration in self.snapshot_iterations:
                self.snapshots[iteration] = (self.w.copy(), float(self.b))
        return self

    def predict(self, X, threshold=0.5):
        """Class prediction: 1 if h(x) >= threshold, otherwise 0."""
        return (self.predict_proba(X) >= threshold).astype(int)


# ----------------------------------------------------------------------------
# 3. Evaluation metrics (written by hand)
# ----------------------------------------------------------------------------
def confusion_counts(y_true, y_pred):
    """Count the four possible outcomes. Positive class = 1 (late arrival)."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))   # late, predicted late
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))   # on time, predicted on time
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))   # on time, predicted late
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))   # late, predicted on time
    return tp, tn, fp, fn


def evaluate(y_true, y_pred):
    """Accuracy, precision, recall and F1 from the confusion counts."""
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ----------------------------------------------------------------------------
# 4. Gradient check: does our calculus agree with a numerical slope?
# ----------------------------------------------------------------------------
def gradient_check(X, y, epsilon=1e-6, seed=0):
    """
    Compare the analytical gradients with central finite differences,
    (J(theta + eps) - J(theta - eps)) / (2 eps), at a random point.
    A tiny difference confirms the derivation and the code match.
    """
    rng = np.random.default_rng(seed)
    model = LogisticRegressionGD()
    model.w = rng.normal(size=X.shape[1])
    model.b = float(rng.normal())
    dw, db = model.compute_gradients(X, y)
    analytical = np.append(dw, db)

    numerical = np.zeros(X.shape[1] + 1)
    for j in range(X.shape[1] + 1):
        original = model.w[j] if j < X.shape[1] else model.b

        def set_param(value):
            if j < X.shape[1]:
                model.w[j] = value
            else:
                model.b = value

        set_param(original + epsilon)
        cost_plus = model.compute_cost(X, y)
        set_param(original - epsilon)
        cost_minus = model.compute_cost(X, y)
        set_param(original)
        numerical[j] = (cost_plus - cost_minus) / (2 * epsilon)
    return analytical, numerical, float(np.max(np.abs(analytical - numerical)))


# ----------------------------------------------------------------------------
# 5. Plot helpers
# ----------------------------------------------------------------------------
def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def scatter_classes(ax, X, y, size=28):
    """Scatter the two classes with different colours and marker shapes."""
    for label in (0, 1):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], s=size, c=CLASS_COLORS[label],
                   marker=CLASS_MARKERS[label], edgecolors="white", linewidths=0.6,
                   alpha=0.9, label=CLASS_NAMES[label])
    ax.set_xlabel(FEATURE_LABELS[0])
    ax.set_ylabel(FEATURE_LABELS[1])
    style_axes(ax)


def boundary_in_original_units(w, b, mu, sigma, x1_values):
    """
    The decision boundary is where h = 0.5, that is where w1*z1 + w2*z2 + b = 0
    (z are the standardized features). Substituting z = (x - mu) / sigma and
    solving for x2 gives a straight line we can draw on the original axes.
    """
    z1 = (x1_values - mu[0]) / sigma[0]
    z2 = -(w[0] * z1 + b) / w[1]
    return z2 * sigma[1] + mu[1]


def draw_boundary(ax, w, b, mu, sigma, xlim, ylim, label="Decision boundary"):
    x1 = np.linspace(xlim[0], xlim[1], 200)
    x2 = boundary_in_original_units(w, b, mu, sigma, x1)
    ax.plot(x1, x2, color="#222222", linewidth=2, label=label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def note_hidden_points(ax, X_raw, xlim):
    """Say how many flights lie beyond the right edge of the plot, so none is silently dropped."""
    hidden = int(np.sum(X_raw[:, 0] > xlim[1]))
    if hidden > 0:
        ax.text(0.99, 0.02, f"{hidden} flights delayed more than {int(xlim[1])} min not shown",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    return hidden


# ----------------------------------------------------------------------------
# 6. Main experiment
# ----------------------------------------------------------------------------
def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # ---- Step 1: data preprocessing -----------------------------------------
    X, y = load_data(DATA_FILE)
    X_train_raw, X_test_raw, y_train, y_test = stratified_train_test_split(X, y, TEST_FRACTION, SEED)
    X_train, X_test, mu, sigma = standardize(X_train_raw, X_test_raw)

    print("=== Step 1: data ===")
    print(f"Samples: {len(y)} | late: {int(y.sum())} | on time: {int((1 - y).sum())}")
    print(f"Train: {len(y_train)} (late {int(y_train.sum())}) | Test: {len(y_test)} (late {int(y_test.sum())})")
    print(f"Training mean: {mu.round(4)} | training std: {sigma.round(4)}")

    # ---- Check the calculus before trusting the training --------------------
    analytical, numerical, max_diff = gradient_check(X_train, y_train)
    print("\n=== Gradient check ===")
    print("Analytical:", analytical)
    print("Numerical: ", numerical)
    print(f"Largest absolute difference: {max_diff:.2e}")

    # ---- Steps 2 and 3: train one model per learning rate --------------------
    models, summary = {}, {}
    print("\n=== Training ===")
    for lr in LEARNING_RATES:
        model = LogisticRegressionGD(learning_rate=lr, n_iterations=N_ITERATIONS,
                                     snapshot_iterations=SNAPSHOT_ITERATIONS)
        model.fit(X_train, y_train)
        models[lr] = model

        test_metrics = evaluate(y_test, model.predict(X_test))
        train_metrics = evaluate(y_train, model.predict(X_train))
        costs = model.cost_history
        summary[str(lr)] = {
            "w": model.w.tolist(), "b": model.b,
            "cost_start": costs[0], "cost_iter_1": costs[1], "cost_iter_10": costs[10],
            "cost_iter_100": costs[100], "cost_final": costs[-1],
            "cost_always_decreasing": bool(np.all(np.diff(costs) < 0)),
            "train_accuracy": train_metrics["accuracy"],
            "test": test_metrics,
        }
        print(f"alpha = {lr:<6} final cost = {costs[-1]:.4f}  w = {model.w.round(4)}  b = {model.b:.4f}  "
              f"test acc = {test_metrics['accuracy']:.4f}  precision = {test_metrics['precision']:.4f}  "
              f"recall = {test_metrics['recall']:.4f}  F1 = {test_metrics['f1']:.4f}")

    # ---- Reference run: how low can the cost go if we train much longer? -------
    reference = LogisticRegressionGD(learning_rate=MAIN_LEARNING_RATE, n_iterations=REFERENCE_ITERATIONS)
    reference.fit(X_train, y_train)
    reference_metrics = evaluate(y_test, reference.predict(X_test))
    print(f"\nReference run ({REFERENCE_ITERATIONS} iterations, alpha = {MAIN_LEARNING_RATE}): "
          f"cost = {reference.cost_history[-1]:.4f}  w = {reference.w.round(4)}  b = {reference.b:.4f}  "
          f"test acc = {reference_metrics['accuracy']:.4f}")

    main_model = models[MAIN_LEARNING_RATE]
    main_metrics = summary[str(MAIN_LEARNING_RATE)]["test"]

    print(f"\n=== Test set performance, alpha = {MAIN_LEARNING_RATE} ===")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"{name:<10}: {main_metrics[name]:.4f}")
    print("Confusion matrix (rows = actual, columns = predicted, order = on time, late):")
    print(np.array([[main_metrics["tn"], main_metrics["fp"]], [main_metrics["fn"], main_metrics["tp"]]]))

    # Shared axis limits so every scatter plot is directly comparable
    xlim = (X[:, 0].min() - 4, X_AXIS_MAX)
    ylim = (X[:, 1].min() - 1, X[:, 1].max() + 1)

    # ---- Figure 2: scatter plot of the training data -------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter_classes(ax, X_train_raw, y_train)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    note_hidden_points(ax, X_train_raw, xlim)
    ax.set_title(f"Training data ({len(y_train)} flights)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIGURE_DIR}/fig2_training_scatter.png", dpi=160); plt.close(fig)

    # ---- Figure 3: cost against iterations for the main model ----------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(main_model.cost_history)), main_model.cost_history, color="#2a78d6", linewidth=2)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Cost J(w, b)")
    ax.set_title(f"Cost during training (learning rate = {MAIN_LEARNING_RATE})")
    style_axes(ax)
    fig.tight_layout(); fig.savefig(f"{FIGURE_DIR}/fig3_cost_curve.png", dpi=160); plt.close(fig)

    # ---- Figure 6: cost curves for the three learning rates ------------------
    lr_colors = {0.1: "#2a78d6", 0.01: "#eb6834", 0.001: "#1baf7a"}
    lr_styles = {0.1: "-", 0.01: "--", 0.001: ":"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr in LEARNING_RATES:
        ax.plot(range(len(models[lr].cost_history)), models[lr].cost_history,
                color=lr_colors[lr], linestyle=lr_styles[lr], linewidth=2, label=f"learning rate = {lr}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Cost J(w, b)")
    ax.set_title("Effect of the learning rate on convergence")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout(); fig.savefig(f"{FIGURE_DIR}/fig6_learning_rates.png", dpi=160); plt.close(fig)

    # ---- Figure 4: final decision boundary on the test data ------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter_classes(ax, X_test_raw, y_test)
    draw_boundary(ax, main_model.w, main_model.b, mu, sigma, xlim, ylim)
    note_hidden_points(ax, X_test_raw, xlim)
    ax.set_title(f"Final decision boundary on the test data ({len(y_test)} flights)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIGURE_DIR}/fig4_final_boundary_test.png", dpi=160); plt.close(fig)

    # ---- Figure 5: the boundary at different stages of training ---------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True, sharey=True)
    for ax, iteration in zip(axes.ravel(), SNAPSHOT_ITERATIONS):
        w_k, b_k = main_model.snapshots[iteration]
        scatter_classes(ax, X_train_raw, y_train, size=16)
        draw_boundary(ax, w_k, b_k, mu, sigma, xlim, ylim)
        note_hidden_points(ax, X_train_raw, xlim)
        ax.set_title(f"After {iteration} iteration{'s' if iteration > 1 else ''}  (cost = {main_model.cost_history[iteration]:.4f})")
    for ax in axes[0]:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"How the decision boundary moves during training (learning rate = {MAIN_LEARNING_RATE})", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{FIGURE_DIR}/fig5_boundary_evolution.png", dpi=160); plt.close(fig)

    # ---- Figure 1: confusion matrix -------------------------------------------
    cm = np.array([[main_metrics["tn"], main_metrics["fp"]], [main_metrics["fn"], main_metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.25)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["On time (0)", "Late (1)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["On time (0)", "Late (1)"])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class")
    cell_names = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{cell_names[i][j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "#222222", fontsize=11)
    ax.set_title("Confusion matrix on the test set")
    fig.tight_layout(); fig.savefig(f"{FIGURE_DIR}/fig1_confusion_matrix.png", dpi=160); plt.close(fig)

    # ---- Save every number used in the report ---------------------------------
    results = {
        "dataset": {"n_samples": int(len(y)), "n_late": int(y.sum()), "n_on_time": int((1 - y).sum()),
                    "n_train": int(len(y_train)), "n_train_late": int(y_train.sum()),
                    "n_test": int(len(y_test)), "n_test_late": int(y_test.sum()),
                    "train_mean": mu.tolist(), "train_std": sigma.tolist()},
        "gradient_check": {"analytical": analytical.tolist(), "numerical": numerical.tolist(),
                           "max_abs_difference": max_diff},
        "learning_rates": summary,
        "reference_run": {"iterations": REFERENCE_ITERATIONS, "cost": reference.cost_history[-1],
                          "w": reference.w.tolist(), "b": reference.b, "test": reference_metrics},
        "snapshots_main_model": {str(k): {"w": v[0].tolist(), "b": v[1],
                                          "cost": main_model.cost_history[k]}
                                 for k, v in main_model.snapshots.items()},
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFigures saved in '{FIGURE_DIR}/' and all numbers saved in results.json")


if __name__ == "__main__":
    main()
