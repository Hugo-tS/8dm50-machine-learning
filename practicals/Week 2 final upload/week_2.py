import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def polynomial_regression_gridsearch(
    X, y,
    max_degree=15,
    test_size=0.25,
    random_state=42,
    cv_splits=5,
    alphas=None,
    plot_cv=True,
    plot_fit=False,
    refit_on_full_for_fit=True,
    n_curve_points=300,
):
    """
    Polynomial regression with Ridge regularization + grid search,
    with optional fitted-line plot (for 1D X).

    Parameters
    ----------
    X : array-like, shape (n_samples,) or (n_samples, n_features)
        Input features.
    y : array-like, shape (n_samples,)
        Targets.
    max_degree : int, default=15
        Max polynomial order to try.
    test_size : float, default=0.25
        Test split fraction.
    random_state : int, default=42
        Reproducibility.
    cv_splits : int, default=5
        K-fold CV splits.
    alphas : list of float, optional
        Ridge strengths to search. If None -> logspace(1e-4..1e3).
    plot_cv : bool, default=True
        Plot CV MSE vs polynomial order (at best alpha).
    plot_fit : bool, default=False
        For 1D X only: plot data + fitted polynomial line.
    refit_on_full_for_fit : bool, default=True
        If True and plot_fit=True, refit pipeline on full (X,y)
        with best hyperparameters before drawing the curve.
    n_curve_points : int, default=300
        Number of x-points for the smooth fitted curve.

    Returns
    -------
    results : dict
        {
          'best_degree': int,
          'best_alpha': float,
          'cv_mse': float,
          'test_mse': float,
          'best_model': fitted Pipeline (train split fit),
          'final_model': fitted Pipeline (full-data refit if plot_fit and refit_on_full_for_fit else None),
          'cv_results_': grid.cv_results_
        }
    """
    # Shape handling
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError(f"X must be 1D or 2D, got shape {X.shape}")
    y = np.asarray(y).ravel()

    # Alpha grid
    if alphas is None:
        alphas = np.logspace(-4, 3, 8)

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

    # Grid: degree x alpha
    param_grid = {
        "poly__degree": list(range(1, max_degree + 1)),
        "ridge__alpha": list(alphas)
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

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

    # Independent test performance 
    y_pred_test = best_model.predict(X_test)
    test_mse    = mean_squared_error(y_test, y_pred_test)

    # Optional plot CV vs degree (at best alpha)
    if plot_cv:
        degrees = np.array(grid.cv_results_["param_poly__degree"], dtype=int)
        alphas_used = np.array(grid.cv_results_["param_ridge__alpha"], dtype=float)
        mean_cv = -grid.cv_results_["mean_test_score"]
        std_cv  = grid.cv_results_["std_test_score"]
        mean_tr = -grid.cv_results_["mean_train_score"]

        mask = alphas_used == best_alpha
        d, cvm, cvs, trm = degrees[mask], mean_cv[mask], std_cv[mask], mean_tr[mask]
        order = np.argsort(d); d, cvm, cvs, trm = d[order], cvm[order], cvs[order], trm[order]

        plt.figure(figsize=(7,4))
        plt.plot(d, cvm, marker="o", label=f"CV MSE (alpha={best_alpha:g})")
        plt.plot(d, trm, marker="s", linestyle="--", label="Train MSE")
        plt.fill_between(d, cvm - cvs, cvm + cvs, alpha=0.15)
        plt.axvline(best_deg, color="gray", linestyle=":", label=f"Best order={best_deg}")
        plt.xlabel("Polynomial order"); plt.ylabel("MSE")
        plt.title("Validation vs Train MSE across polynomial order (Ridge)")
        plt.legend(); plt.grid(True); plt.show()

    # Optional fitted curve plot (1D only)
    final_model = None
    if plot_fit:
        if X.shape[1] != 1:
            print("plot_fit=True ignored: fitted curve is only defined for 1D features.")
        else:
            if refit_on_full_for_fit:
                # refit with best params on ALL data for the nicest curve
                final_model = Pipeline([
                    ("poly",   PolynomialFeatures(degree=best_deg, include_bias=False)),
                    ("scaler", StandardScaler()),
                    ("ridge",  Ridge(alpha=best_alpha, random_state=random_state)),
                ]).fit(X, y)
                model_for_curve = final_model
            else:
                model_for_curve = best_model  # trained on train split only

            # smooth x-grid
            x_min, x_max = X.min(), X.max()
            X_plot = np.linspace(x_min, x_max, n_curve_points).reshape(-1, 1)
            y_plot = model_for_curve.predict(X_plot)

            # scatter full data + fitted line
            plt.figure(figsize=(6,4))
            plt.scatter(X, y, color="red", s=20, alpha=0.7, label="Data")
            plt.plot(X_plot, y_plot, color="blue", linewidth=2, label=f"Polynomial fit (order={best_deg}, α={best_alpha:g})")
            plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.grid(True); plt.show()

    return {
        "best_degree": best_deg,
        "best_alpha": best_alpha,
        "cv_mse": cv_mse,
        "test_mse": test_mse,
        "best_model": best_model,
        "final_model": final_model,
        "cv_results_": grid.cv_results_,
    }

def plot_knn_roc_curves(X, y, k_values=range(1, 15, 2), test_size=0.25, random_state=None):
    """
    Train k-NN classifiers with different k values and plot their ROC curves.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature data.
    y : array-like of shape (n_samples,)
        Binary target values.
    k_values : iterable of int, default=range(1, 15, 2)
        Values of k (number of neighbors) to evaluate.
    test_size : float, default=0.25
        Proportion of the dataset to include in the test split.
    random_state : int, default=None
        Random state for reproducibility.

    Returns
    -------
    None
        Displays the ROC plot.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    plt.figure()
    plt.title("k-NN ROC curves on Breast Cancer (test set)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    for k in k_values:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])

        model.fit(X_train, y_train)

        # Predict probabilities for the positive class
        y_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.plot(fpr, tpr, label=f"k={k} (AUC={auc:.3f})")

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()