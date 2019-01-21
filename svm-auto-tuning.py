## Python version

"""
Import packages
"""
import sys
import scipy
import numpy as np
np.set_printoptions(threshold=np.inf)


import pandas as pd
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn.metrics import classification_report, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


"""
Some intro Data reading 
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

s_sensitivity = []
s_specificity = []
s_precision = []
svc_test_accuracy = []
svc_train_accuracy = []
svc_test_auc = []
svc_train_auc = []

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
    svm = SVC(random_state=42)

    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    shrinking = [True, False]
    degree = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    param_grid = {'C': Cs, 'gamma': gammas, 'shrinking': shrinking, 'degree': degree}
    grid_search = GridSearchCV(svm, param_grid, cv=None)
    best_model = grid_search.fit(training_inputs, training_classes)
    print("Best: %f using %s" % (best_model.best_score_, best_model.best_params_))

    """
    SVM
    """
    predictions = best_model.predict(testing_inputs)
    trainpredictions = best_model.predict(training_inputs)

    svc_test_accuracy.append(metrics.accuracy_score(testing_classes, predictions))
    svc_train_accuracy.append(metrics.accuracy_score(training_classes, trainpredictions))

    fpr, tpr, _ = metrics.roc_curve(testing_classes, predictions)
    svc_test_auc.append(metrics.auc(fpr, tpr))
    fpr2, tpr2, _ = metrics.roc_curve(training_classes, trainpredictions)
    svc_train_auc.append(metrics.auc(fpr2, tpr2))

    report = classification_report(testing_classes, predictions, output_dict=True)
    sensitivity = report['1']['recall']
    specificity = report['0']['recall']
    precision = precision_score(testing_classes, predictions)

    s_sensitivity.append(sensitivity)
    s_specificity.append(specificity)
    s_precision.append(precision)

"""
Print average
"""

# SVM data

print ("svm train auc is: ", np.mean(svc_train_auc))
print("svm test auc is: ", np.mean(svc_test_auc))

print ("svm train accuracy is: ", np.mean(svc_train_accuracy))
print("svm test accuracy is: ", np.mean(svc_test_accuracy))

print("")

print("SVM info:")
print("sensitivity: ", np.mean(s_sensitivity))
print("specificity: ", np.mean(s_specificity))
print("precision: ", np.mean(s_precision))



