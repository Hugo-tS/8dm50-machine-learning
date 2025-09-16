import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def polynomial_regression_gridsearch(
    X, y, max_degree=15, test_size=0.25, random_state=42,
    cv_splits=5, plot=True,
    alphas=None
):
    """
    Perform polynomial regression with grid search over polynomial degree.
    
    Parameters
    ----------
    X : array-like, shape (n_samples,)
        Input feature values (1D).
    y : array-like, shape (n_samples,)
        Target values (1D).
    max_degree : int, default=15
        Maximum polynomial degree to test.
    test_size : float, default=0.25
        Fraction of samples used for the test set.
    random_state : int, default=42
        Seed for reproducibility.
    cv_splits : int, default=5
        Number of folds for cross-validation.
    plot : bool, default=True
        If True, plot validation vs. train MSE as a function of degree.
    alphas : float, optional
        Sequence of Ridge regularization strengths to search over.
        If None, defaults to a log-spaced range [1e-4, ..., 1e3].
    
    Returns
    -------
    dict with:
        - best_degree
        - cv_mse
        - test_mse
        - best_model (fitted pipeline)
    """

    # Shape handling
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError(f"X must be 1D or 2D, got shape {X.shape}")
    y = np.asarray(y).ravel()

    # default alpha grid (log-spaced)
    if alphas is None:
        alphas = np.logspace(-4, 3, 8)   # 1e-4 ... 1e3

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Pipeline
    pipe = Pipeline([
        ("poly",   PolynomialFeatures(include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(random_state=random_state))
    ])

    # search both degree and alpha
    param_grid = {
        "poly__degree": list(range(1, max_degree + 1)),
        "ridge__alpha": list(alphas)
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # Grid Search
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        return_train_score=True,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_deg   = grid.best_params_["poly__degree"]
    best_alpha = grid.best_params_["ridge__alpha"]
    cv_mse     = -grid.best_score_
    best_model = grid.best_estimator_

    y_pred_test = best_model.predict(X_test)
    test_mse    = mean_squared_error(y_test, y_pred_test)

    # Plot
    if plot:
        degrees = np.array(grid.cv_results_["param_poly__degree"], dtype=int)
        alphas_used = np.array(grid.cv_results_["param_ridge__alpha"], dtype=float)
        mean_cv = -grid.cv_results_["mean_test_score"]
        std_cv  = grid.cv_results_["std_test_score"]
        mean_tr = -grid.cv_results_["mean_train_score"]

        # show the curve versus degree at the best alpha (for readability)
        mask = alphas_used == best_alpha
        d, cvm, cvs, trm = degrees[mask], mean_cv[mask], std_cv[mask], mean_tr[mask]
        order = np.argsort(d)
        d, cvm, cvs, trm = d[order], cvm[order], cvs[order], trm[order]

        plt.figure(figsize=(7,4))
        plt.plot(d, cvm, marker="o", label=f"CV MSE (alpha={best_alpha:g})")
        plt.plot(d, trm, marker="s", linestyle="--", label="Train MSE")
        plt.fill_between(d, cvm - cvs, cvm + cvs, alpha=0.15)
        plt.axvline(best_deg, color="gray", linestyle=":", label=f"Best deg={best_deg}")
        plt.xlabel("Polynomial degree")
        plt.ylabel("MSE")
        plt.title("Validation vs Train MSE across polynomial degree (Ridge)")
        plt.legend(); plt.grid(True); plt.show()

    return {
        "best_degree": best_deg,
        "best_alpha": best_alpha,
        "cv_mse": cv_mse,
        "test_mse": test_mse,
        "best_model": best_model,
        "cv_results_": grid.cv_results_,
    }