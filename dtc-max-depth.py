## Python version

"""
Import packages
"""

import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from matplotlib.legend_handler import HandlerLine2D

"""
Some intro Data reading and description
"""
data = pd.read_csv('norm-reduced.csv')


"""
Identifies all inputs and all outputs
"""

all_inputs = (data.drop(['Remission_Type'], axis=1)).values
all_labels = data['Remission_Type'].values

"""
K-fold validation
"""
from sklearn.model_selection import KFold

dtc_accuracy = []
dtc_train_accuracy = []
dtc_auc = []
dtc_train_auc = []

kf = KFold(n_splits=10, random_state=42)

best_features = {}
for train_index, test_index in kf.split(all_inputs):


    """
    Defines training and testing
    """
    training_inputs, testing_inputs = all_inputs[train_index], all_inputs[test_index]
    training_classes, testing_classes = all_labels[train_index], all_labels[test_index]

    """
    DTC: Recursive Feature Elimination
    """

    dtc = DecisionTreeClassifier(max_depth=3, min_samples_split=3, random_state=42)
    rfe = RFE(dtc, n_features_to_select=50, verbose=0)
    fit_dtc = rfe.fit(training_inputs, training_classes)

    """
    feature selection
    """
    low_rankings = []
    for i in range(fit_dtc.support_.size):
        if fit_dtc.support_[i]:
            low_rankings.append(i)
    training_inputs = training_inputs[:, low_rankings]
    testing_inputs = testing_inputs[:, low_rankings]

    """
    Decision Tree classifier: Testing AUC for various values of max_depth
    """

    # iterates over a loop for max_depth
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []
    for max_depth in max_depths:

        # Initializes DTC
        dtc = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
        dtc.fit(training_inputs, training_classes)

        # Training data
        trainpredictions = dtc.predict(training_inputs)
        fpr2, tpr2, _ = metrics.roc_curve(training_classes, trainpredictions)
        dtc_train_accuracy.append(metrics.accuracy_score(training_classes, trainpredictions))
        dtc_train_auc.append(metrics.auc(fpr2, tpr2))
        train_results.append(metrics.auc(fpr2, tpr2))

        # Testing data
        testpredictions = dtc.predict(testing_inputs)
        dtc_accuracy.append(metrics.accuracy_score(testing_classes, testpredictions))
        fpr, tpr, _ = metrics.roc_curve(testing_classes, testpredictions)
        dtc_auc.append(metrics.auc(fpr, tpr))
        test_results.append(metrics.auc(fpr, tpr))

    """
    Print AUC graph or different max_depths
    """
    line1, = plt.plot(max_depths, train_results, 'b', label="Decision Tree Train AUC")
    line2, = plt.plot(max_depths, test_results, 'r', label="Decision Tree Test AUC")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('Auc Score')
    plt.xlabel('Tree Depth')
    plt.show()
