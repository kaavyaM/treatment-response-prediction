## Python version

"""
Most of these packages that we haven't used yet but might want to use in the future.
"""
import sys
import scipy
import numpy as np
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from matplotlib.legend_handler import HandlerLine2D
from sklearn.neighbors import KNeighborsClassifier

"""
Some intro Data reading and description
"""
data = pd.read_csv('preprocessed.csv')

"""
Identifies all inputs and all outputs
"""

all_inputs = (data.drop(['Remission_Type'], axis=1)).values
all_labels = data['Remission_Type'].values

"""
K-fold validation
"""
from sklearn.model_selection import KFold

# Train and test dat

knn_test_accuracy = []
knn_train_accuracy = []
knn_test_auc = []
knn_train_auc = []

kf = KFold(n_splits=5, random_state=42)

for train_index, test_index in kf.split(all_inputs):

    """
    Defines training and testing
    """
    training_inputs, testing_inputs = all_inputs[train_index], all_inputs[test_index]
    training_classes, testing_classes = all_labels[train_index], all_labels[test_index]

    """
    K Nearest Neighbors
    """

    distances = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_results = []
    test_results = []

    for p in distances:
        knn = KNeighborsClassifier(p=p)
        knn = knn.fit(training_inputs, training_classes)
        trainpredictions = knn.predict(training_inputs)
        testpredictions = knn.predict(testing_inputs)

        knn_test_accuracy.append(metrics.accuracy_score(testing_classes, testpredictions))
        knn_train_accuracy.append(metrics.accuracy_score(training_classes, trainpredictions))

        fpr, tpr, _ = metrics.roc_curve(testing_classes, testpredictions)
        knn_test_auc.append(metrics.auc(fpr, tpr))
        fpr2, tpr2, _ = metrics.roc_curve(training_classes, trainpredictions)
        knn_train_auc.append(metrics.auc(fpr2, tpr2))

        train_results.append(metrics.auc(fpr2, tpr2))
        test_results.append(metrics.auc(fpr, tpr))

    line1, = plt.plot(distances, train_results, 'b', label="Train AUC")
    line2, = plt.plot(distances, test_results, 'r', label="Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Auc Score')
    plt.xlabel('p')
    plt.show()
