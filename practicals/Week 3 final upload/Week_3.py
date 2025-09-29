import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score


def coef_path(model_class, alphas, X, y):
    """
    Computes coefficient paths for a range of alpha values using a linear model.

    Parameters
    ----------
    model_class : sklearn estimator class
        A linear model class that accepts `alpha` and `max_iter` as arguments (e.g., Lasso, Ridge).
    alphas : list or array-like
        Sequence of alpha (regularization strength) values to test.
    X : ndarray or DataFrame
        Feature matrix.
    y : ndarray or Series
        Target vector.

    Returns
    -------
    np.ndarray
        Array of coefficient vectors, one for each alpha.
    """
    coefs = []
    for a in alphas:
        # Build a pipeline: standardize features before fitting the model
        pipe = make_pipeline(StandardScaler(), model_class(alpha=a, max_iter=100000))
        pipe.fit(X, y)
        # Extract model coefficients
        coefs.append(pipe[-1].coef_)
    return np.array(coefs)


def bootstrap_coef_stats(model_cls, alphas, X, y, B):
    """
    Estimates mean and standard deviation of model coefficients via bootstrapping.

    Parameters
    ----------
    model_cls : sklearn estimator class
        Linear model class (e.g., Ridge, Lasso).
    alphas : list or array-like
        Sequence of alpha values to test.
    X : ndarray or DataFrame
        Feature matrix.
    y : ndarray or Series
        Target vector.
    B : int
        Number of bootstrap resamples.

    Returns
    -------
    tuple (coef_mean, coef_std)
        - coef_mean : np.ndarray
            Mean coefficients across bootstrap samples for each alpha.
        - coef_std : np.ndarray
            Standard deviation of coefficients across bootstrap samples for each alpha.
    """
    coef_mean = []
    coef_std = []
    for a in alphas:
        boot_coefs = []
        for _ in range(B):
            Xb, yb = resample(X, y, replace=True, random_state=None)
            pipe = make_pipeline(StandardScaler(), model_cls(alpha=a, max_iter=100_000))
            pipe.fit(Xb, yb)
            boot_coefs.append(pipe[-1].coef_)
        boot_coefs = np.array(boot_coefs)
        coef_mean.append(boot_coefs.mean(axis=0))
        coef_std.append(boot_coefs.std(axis=0))
    return np.array(coef_mean), np.array(coef_std)

def cv_curve(model_cls, alphas, X, y, cv):
    """
    Computes cross-validation errors for a range of alpha values.

    Parameters
    ----------
    model_cls : sklearn estimator class
        Linear model class (e.g., Ridge, Lasso) that accepts `alpha` and `max_iter`.
    alphas : list or array-like
        Sequence of alpha (regularization strength) values to test.
    X : ndarray or DataFrame
        Feature matrix.
    y : ndarray or Series
        Target vector.
    cv : int or cross-validation generator
        Number of folds (e.g., 5 or 10) or a CV splitter object.

    Returns
    -------
    np.ndarray
        Array of mean cross-validated errors (MSE) for each alpha.
        Lower values indicate better model performance.
    """

    scores = []
    for a in alphas:
        # Build a pipeline: standardize features before fitting the model
        pipe = make_pipeline(StandardScaler(), model_cls(alpha=a, max_iter=100000))
        # Use negative MSE as scoring; negate to get positive MSE
        s = cross_val_score(pipe, X, y, cv=cv, scoring='neg_mean_squared_error')
        # Average the scores across folds
        scores.append(-s.mean())
    return np.array(scores)