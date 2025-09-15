import numpy as np

class KNeighborsClassifier():
    """
    Simple k-Nearest Neighbors classifier (Euclidean).

    Features:
      - Optional z-score normalization using train statistics only.
      - Vectorized pairwise distance computation.
      - Majority vote where smallest label wins in case of a tie.

    Methods:
      - fit(X, y): memorize training data (+ normalization stats)
      - predict(X, k): predict labels for queries X using k neighbors
      - score(X, y, k): accuracy on (X, y) for a given k
    """
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.X_train_ = None
        self.y_train_ = None
        self.mu_ = None
        self.sigma_ = None
        self.classes_ = None

    # normalization helpers 
    @staticmethod
    def _zscore_fit(X):
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, ddof=0, keepdims=True)
        sigma[sigma == 0.0] = 1.0  # avoid division by zero
        return mu, sigma

    @staticmethod
    def _zscore_transform(X, mu, sigma):
        return (X - mu) / sigma

    # Distances 
    @staticmethod
    def Euclidean_distance_metric(XA, XB):
        """
        Compute Euclidean distance matrix between rows of XA and XB.
        XA: (n_a, d)
        XB: (n_b, d)
        returns: D (n_a, n_b)
        """
        AA = np.sum(XA**2, axis=1, keepdims=True)        # (n_a, 1)
        BB = np.sum(XB**2, axis=1, keepdims=True).T      # (1, n_b)
        AB = XA @ XB.T                                   # (n_a, n_b)
        D2 = AA + BB - 2.0 * AB
        D2[D2 < 0] = 0.0
        D = np.sqrt(D2)
        return D

    # public API 
    def fit(self, X, y):
        """
        Memorize training data (with optional normalization).
        X: (n_train, d)
        y: (n_train,) or (n_train, 1)
        """
        y = np.asarray(y).ravel().astype(int)
        X = np.asarray(X, dtype=float)
        if self.normalize:
            self.mu_, self.sigma_ = self._zscore_fit(X)
            Xn = self._zscore_transform(X, self.mu_, self.sigma_)
        else:
            self.mu_, self.sigma_ = None, None
            Xn = X

        self.X_train_ = Xn
        self.y_train_ = y
        self.classes_ = np.unique(y)
        return self

    def _prepare_queries(self, X):
        X = np.asarray(X, dtype=float)
        if self.normalize:
            if self.mu_ is None or self.sigma_ is None:
                raise ValueError("Model not fitted yet.")
            return self._zscore_transform(X, self.mu_, self.sigma_)
        return X

    def predict(self, X, k=5):
        """
        Predict labels for queries X using k nearest neighbors.
        X: (n_test, d)
        returns: y_pred (n_test,)
        """
        if self.X_train_ is None:
            raise ValueError("Call fit() before predict().")
        if k < 1 or k > len(self.X_train_):
            raise ValueError(f"k must be in [1, {len(self.X_train_)}], got {k}.")

        Xq = self._prepare_queries(X)                           # (n_test, d)
        D2 = self.Euclidean_distance_metric(Xq, self.X_train_)      # (n_test, n_train)

        # indices of k nearest neighbors for each query
        nn_idx = np.argpartition(D2, kth=k-1, axis=1)[:, :k]    # (n_test, k)
        nn_labels = self.y_train_[nn_idx]                       # (n_test, k)

        # majority vote 
        n_test = nn_labels.shape[0]
        y_pred = np.empty(n_test, dtype=int)

        # to make bincount stable across rows, set minlength to number of classes if small
        n_classes = int(self.classes_.max()) + 1

        for i in range(n_test):
            counts = np.bincount(nn_labels[i], minlength=n_classes)
            y_pred[i] = counts.argmax()  # tie -> smallest label

        return y_pred

    def evaluate(self, X, y, k=5):
        """
        Accuracy on (X, y) using k.
        """
        y = np.asarray(y).ravel().astype(int)
        y_pred = self.predict(X, k=k)
        return np.mean(y_pred == y)
    
    import numpy as np

class KNeighborsRegressor:
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.X_train_ = None
        self.y_train_ = None
        self.mu_ = None
        self.sigma_ = None

    # same helpers as in classifier
    @staticmethod
    def _zscore_fit(X):
        mu = X.mean(axis=0, keepdims=True)
        sigma = X.std(axis=0, ddof=0, keepdims=True)
        sigma[sigma == 0.0] = 1.0
        return mu, sigma

    @staticmethod
    def _zscore_transform(X, mu, sigma):
        return (X - mu) / sigma

    @staticmethod
    def _pairwise_sqeuclidean(XA, XB):
        AA = np.sum(XA**2, axis=1, keepdims=True)
        BB = np.sum(XB**2, axis=1, keepdims=True).T
        AB = XA @ XB.T
        D2 = AA + BB - 2.0 * AB
        D2[D2 < 0] = 0.0
        return D2

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()   # keep as float!
        if self.normalize:
            self.mu_, self.sigma_ = self._zscore_fit(X)
            self.X_train_ = self._zscore_transform(X, self.mu_, self.sigma_)
        else:
            self.mu_, self.sigma_ = None, None
            self.X_train_ = X
        self.y_train_ = y
        return self

    def _prepare_queries(self, X):
        X = np.asarray(X, float)
        if self.normalize:
            return self._zscore_transform(X, self.mu_, self.sigma_)
        return X

    def predict(self, X, k=5, weighted=False, eps=1e-12):
        if self.X_train_ is None:
            raise ValueError("Call fit() before predict().")
        if k < 1 or k > len(self.X_train_):
            raise ValueError(f"k must be in [1, {len(self.X_train_)}]")

        Xq = self._prepare_queries(X)
        D2 = self._pairwise_sqeuclidean(Xq, self.X_train_)        # (n_test, n_train)
        nn_idx = np.argpartition(D2, kth=k-1, axis=1)[:, :k]      # (n_test, k)
        nn_y = self.y_train_[nn_idx]                              # (n_test, k)

        if not weighted:
            return nn_y.mean(axis=1)                              # unweighted mean
        else:
            d = np.sqrt(np.take_along_axis(D2, nn_idx, axis=1))   # (n_test, k)
            w = 1.0 / (d + eps)
            w /= w.sum(axis=1, keepdims=True)
            return (w * nn_y).sum(axis=1)

    def mse(self, X, y, k=5, **pred_kwargs):
        y = np.asarray(y, float).ravel()
        yhat = self.predict(X, k=k, **pred_kwargs)
        return np.mean((yhat - y)**2)
