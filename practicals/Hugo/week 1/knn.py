import numpy as np
from scipy.spatial import distance

def knn_predict(X_train, y_train, X_test, k):
    """
    k-NN prediction
    """
    #Compute distance between each test point and all train points
    dists = distance.cdist(X_test, X_train, 'euclidean')  #shape: (n_test, n_train)
    
    y_pred = []
    for i in range(dists.shape[0]):
        #Get indices of knn
        neighbors_index = np.argsort(dists[i])[:k]
        neighbors_labels = y_train[neighbors_index]
        
        #Majority vote
        counts = np.bincount(neighbors_labels)
        y_pred.append(np.argmax(counts))
    return np.array(y_pred)


def knn_regression_predict(X_train, y_train, X_test, k):
    """
    k-NN regression prediction
    """
    #Compute distances between test and train points
    dists = distance.cdist(X_test, X_train, 'euclidean')   #shape: (n_test, n_train)

    y_pred = []
    for i in range(dists.shape[0]):
        # Get indices of knn
        neighbors_index = np.argsort(dists[i])[:k]
        neighbors_values = y_train[neighbors_index]

        # Average instead of majority vote
        y_pred.append(np.mean(neighbors_values))
    return np.array(y_pred)

