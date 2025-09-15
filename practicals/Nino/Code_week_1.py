import numpy as np

def lsq(X_train, y_train):
    XtX = X_train.T @ X_train
    XtX_inv = np.linalg.inv(XtX)
    weights = XtX_inv @ X_train.T @ y_train
    return weights


def predict(X, weights):
    y_predicted = X @ weights
    return y_predicted


def mean_squared_error(y_predicted, y_true):
    errors = y_true - y_predicted
    mse = np.mean(errors ** 2)
    return mse


def kNN_analyser(k, X_train, y_train, X_test, y_test):
    instances, features = X_test.shape
    y_predicted = []

    for x in range(0, instances):
        instance = X_test[x, :]
        prediction = kNN_predictor(k, X_train, y_train, instance)
        y_predicted.append(prediction)
    
    accuracy = np.mean(y_predicted == y_test)

    return accuracy



def kNN_predictor(k, X_train, y_train, instance):
    distances = np.zeros((X_train.shape[0], 2))
    distances[:, 1] = y_train[:,0]
    for x in range(0, X_train.shape[0]):
        data_point = X_train[x,:]
        euc_distance = np.sqrt(np.sum((data_point-instance)**2))
        distances[x,0] = euc_distance

    idx = np.argsort(distances[:, 0])          
    distances_sorted = distances[idx]   
    y = distances_sorted[0:k, 1]
    number_of_zero = np.sum(y == 0)
    number_of_ones = np.sum(y == 1)

    # Wat als het gelijk is? Nu wordt het dan prediction = 1. Of alleen oneven k gebruiken?
    if number_of_zero > number_of_ones:
        prediction = 0
    else:
        prediction = 1
    return prediction
    



def kNN_regression_analyser(k, X_train, y_train, X_test, y_test):
    instances, features = X_test.shape
    y_predicted = []

    for x in range(0, instances):
        instance = X_test[x, :]
        prediction = kNN_regression_predictor(k, X_train, y_train, instance)
        y_predicted.append(prediction)
    
    errors = y_test - y_predicted
    mse = np.mean(errors ** 2)
    return mse

def kNN_regression_predictor(k, X_train, y_train, instance):
    distances = np.zeros((X_train.shape[0], 2))
    distances[:, 1] = y_train[:,0]
    for x in range(0, X_train.shape[0]):
        data_point = X_train[x,:]
        euc_distance = np.sqrt(np.sum((data_point-instance)**2))
        distances[x,0] = euc_distance

    idx = np.argsort(distances[:, 0])          
    distances_sorted = distances[idx]   
    y = distances_sorted[0:k, 1]
    prediction = y.mean()

    return prediction