## Python version

"""
Import packages
"""

import numpy as np
np.set_printoptions(threshold=np.inf)

import pandas as pd
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import classification_report, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
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

k_sensitivity = []
k_specificity = []
k_precision = []


kf = KFold(n_splits=10, random_state=42)

best_features = {}


for train_index, test_index in kf.split(all_inputs):

    """
    Defines training and testing
    """
    training_inputs, testing_inputs = all_inputs[train_index], all_inputs[test_index]
    training_classes, testing_classes = all_labels[train_index], all_labels[test_index]

    """
    Logistic: Recursive Feature elimination
    """

    logisticRegr = LogisticRegression(max_iter=1000)
    rfe = RFE(logisticRegr, n_features_to_select=50, verbose=0)
    fit_logistic = rfe.fit(training_inputs, training_classes)

    """
    feature selection
    """
    low_rankings = []
    for i in range(fit_logistic.support_.size):
        if fit_logistic.support_[i]:
            low_rankings.append(i)
    training_inputs = training_inputs[:, low_rankings]
    testing_inputs = testing_inputs[:, low_rankings]


    """
    Hyperparameter tuning
    """
    knn = KNeighborsClassifier()

    n_neighbors = [1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    weights = ['uniform', 'distance']
    algorithm = ['auto']
    leaf_size = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]
    p = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    param_grid = dict(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
    random = RandomizedSearchCV(knn, param_grid, n_jobs=-1, cv=2)
    best_model = random.fit(training_inputs, training_classes)
    print("Best: %f using %s" % (best_model.best_score_, best_model.best_params_))

    """
    K Nearest Neighbors
    """
    trainpredictions = best_model.predict(training_inputs)
    testpredictions = best_model.predict(testing_inputs)

    knn_test_accuracy.append(metrics.accuracy_score(testing_classes, testpredictions))
    knn_train_accuracy.append(metrics.accuracy_score(training_classes, trainpredictions))

    fpr, tpr, _ = metrics.roc_curve(testing_classes, testpredictions)
    knn_test_auc.append(metrics.auc(fpr, tpr))
    fpr2, tpr2, _ = metrics.roc_curve(training_classes, trainpredictions)
    knn_train_auc.append(metrics.auc(fpr2, tpr2))

    report = classification_report(testing_classes, testpredictions, output_dict=True)
    sensitivity = report['1']['recall']
    specificity = report['0']['recall']
    precision = precision_score(testing_classes, testpredictions)

    k_sensitivity.append(sensitivity)
    k_specificity.append(specificity)
    k_precision.append(precision)

"""
Print average
"""

# KNN data

print ("knn train auc is: ", np.mean(knn_train_auc))
print("knn test auc is: ", np.mean(knn_test_auc))

print ("knn train accuracy is: ", np.mean(knn_train_accuracy))
print("knn test accuracy is: ", np.mean(knn_test_accuracy))

print("")

print("KNN info:")
print("sensitivity: ", np.mean(k_sensitivity))
print("specificity: ", np.mean(k_specificity))
print("precision: ", np.mean(k_precision))

