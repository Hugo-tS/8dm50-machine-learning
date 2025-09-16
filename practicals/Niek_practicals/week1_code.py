
import numpy as np
import matplotlib.pyplot as plt

def normalize_data(X_train: np.ndarray, X_test: np.ndarray):
    """
    Min–Max normalize features to [0,1] using only training-set statistics.
    Returns:
        X_train_norm, X_test_norm, Xmin, Xden
    Where Xden is the (Xmax - Xmin) with ones substituted for zeros to avoid dividing by 0.
    """
    Xmin = X_train.min(axis=0, keepdims=True)
    Xmax = X_train.max(axis=0, keepdims=True)
    den = np.where((Xmax - Xmin) == 0, 1.0, (Xmax - Xmin))
    X_train_norm = (X_train - Xmin) / den
    X_test_norm  = (X_test  - Xmin) / den
    return X_train_norm, X_test_norm, Xmin, den


def squared_euclidian(A, B):
    """
    Compute all squared Euclidean distances between rows of A (m x d) and B (n x d).
    Returns a (m x n) matrix where entry (i, j) = ||A[i] - B[j]||^2.
    """
    A_sq = np.sum(A**2, axis=1, keepdims=True)        # (m, 1)
    B_sq = np.sum(B**2, axis=1, keepdims=True).T      # (1, n)
    cross = A @ B.T                                   # (m, n)
    d2 = A_sq + B_sq - 2.0 * cross
    # Numerical guard: clip tiny negative values to zero
    return np.maximum(d2, 0.0)


def prepare_neighbors(X_train_norm: np.ndarray, y_train: np.ndarray, X_test_norm: np.ndarray):
    """
    Precompute:
        - neighbor_order: indices of training samples sorted by distance per test sample
        - sorted_labels: y_train reordered by neighbor_order
        - cum_ones: cumulative sum of 1s across neighbors for fast majority voting
    """
    d2 = squared_euclidian(X_test_norm, X_train_norm)  # (n_test, n_train)
    neighbor_order = np.argsort(d2, axis=1)                     # (n_test, n_train)
    y_train = y_train.astype(int)
    sorted_labels = y_train[neighbor_order]                     # (n_test, n_train)
    cum_ones = np.cumsum(sorted_labels, axis=1)                 # (n_test, n_train)
    return neighbor_order, sorted_labels, cum_ones

def evaluate_k_list(cum_ones: np.ndarray, y_test: np.ndarray, k_list):
    """
    Vectorized evaluation of accuracy over multiple k values.
    cum_ones: (n_test, n_train) cumulative ones among sorted neighbors
    y_test:   (n_test,)
    k_list:   iterable of positive integers <= n_train
    Returns: (ks, accuracies) as lists
    """
    y_test = y_test.astype(int)
    accuracies = []
    for k in k_list:
        ones_in_k = cum_ones[:, k-1]                 # # of ones among top-k
        preds = (ones_in_k > (k / 2)).astype(int)    # majority vote (ties -> 0)
        acc = np.mean(preds == y_test)
        accuracies.append(acc)
    return list(k_list), accuracies


def plot_k_vs_accuracy(ks, accuracies, title="kNN accuracy vs k"):
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, accuracies, marker='o')
    plt.xlabel("k (number of neighbors)")
    plt.ylabel("Accuracy on test split")
    plt.title(title)
    plt.grid(True)
    plt.show()

def knn_regression_predict(X_train_norm: np.ndarray, y_train: np.ndarray,
                           X_test_norm: np.ndarray, k: int) -> np.ndarray:
    if k < 1 or k > X_train_norm.shape[0]:
        raise ValueError("k must be in [1, n_train].")
    d2 = squared_euclidian(X_test_norm, X_train_norm)   # (n_test, n_train)
    order = np.argsort(d2, axis=1)[:, :k]                        # top-k neighbor indices
    return np.mean(y_train[order], axis=1)                       # average neighbors' targets

def linear_regression_fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                                  X_test: np.ndarray) -> np.ndarray:
    Xtr = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # add bias
    Xte = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    beta, *_ = np.linalg.lstsq(Xtr, y_train, rcond=None)  # stable least-squares
    return Xte @ beta

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)




def gaussian_params_by_class(X: np.ndarray, y: np.ndarray):
    """
    Return:
        means: (n_classes, n_features)
        vars_: (n_classes, n_features)
    Uses MLE variance (ddof=0). Adds tiny epsilon to avoid zero-variance issues.
    """
    classes = np.unique(y)
    means = np.zeros((classes.size, X.shape[1]), dtype=float)
    vars_ = np.zeros_like(means)

    for ci, c in enumerate(classes):
        Xc = X[y == c]
        means[ci] = Xc.mean(axis=0)
        vars_[ci] = Xc.var(axis=0)  # MLE (divide by N)
    eps = 1e-12
    vars_ = np.maximum(vars_, eps)
    return means, vars_

def normal_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    coef = 1.0 / np.sqrt(2.0 * np.pi * var)
    return coef * np.exp(-0.5 * (x - mu) ** 2 / var)


def plot_class_conditionals_all_features(X, y, means, vars_, feature_names, class_names):
    n_features = X.shape[1]
    nrows, ncols = 6, 5                            # 6x5 grid for 30 features
    assert nrows * ncols == n_features

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 16), sharex=False, sharey=False)
    axes = axes.ravel()

    # We'll capture handles from the first axes for a global legend
    legend_handles = None

    for j in range(n_features):
        ax = axes[j]
        xj = X[:, j]
        xmin, xmax = xj.min(), xj.max()
        pad = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
        xs = np.linspace(xmin - pad, xmax + pad, 300)

        # Class 0 and 1 PDFs
        p0 = normal_pdf(xs, means[0, j], vars_[0, j])
        p1 = normal_pdf(xs, means[1, j], vars_[1, j])

        l0, = ax.plot(xs, p0, label=class_names[0])
        l1, = ax.plot(xs, p1, label=class_names[1])

        if legend_handles is None:
            legend_handles = (l0, l1)

        ax.set_title(feature_names[j], fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("density")

    # Global title and legend
    fig.suptitle("Breast Cancer: Class-Conditional Densities per Feature\nAssuming Gaussian p(X_j | Y)", fontsize=16)
    if legend_handles is not None:
        fig.legend(legend_handles, class_names, loc="upper center", ncol=2, frameon=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()