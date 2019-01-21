## Python version

"""
Most of these packages that we haven't used yet but might want to use in the future.
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

logistic_test_accuracy = []
logistic_train_accuracy = []
logistic_test_auc = []
logistic_train_auc = []

# Classification report
l_sensitivity = []
l_specificity = []
l_precision = []

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
    # #
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

    penalty = ['l2']
    dual = [True, False]
    max_iter = [500, 700, 900, 1100]
    C = [1.0, 1.5, 2.0, 2.5]

    param_grid = dict(dual=dual, max_iter=max_iter, C=C, penalty=penalty)
    random = GridSearchCV(logisticRegr, param_grid, n_jobs=-1, cv=2)
    best_model = random.fit(training_inputs, training_classes)
    print("Best: %f using %s" % (best_model.best_score_, best_model.best_params_))

    """
    Logistic
    """

    trainpredictions = best_model.predict(training_inputs)
    testpredictions = best_model.predict(testing_inputs)

    logistic_test_accuracy.append(metrics.accuracy_score(testing_classes, testpredictions))
    logistic_train_accuracy.append(metrics.accuracy_score(training_classes, trainpredictions))

    fpr, tpr, _ = metrics.roc_curve(testing_classes, testpredictions)
    logistic_test_auc.append(metrics.auc(fpr, tpr))
    fpr2, tpr2, _ = metrics.roc_curve(training_classes, trainpredictions)
    logistic_train_auc.append(metrics.auc(fpr2, tpr2))

    report = classification_report(testing_classes, testpredictions, output_dict=True)
    sensitivity = report['1']['recall']
    specificity = report['0']['recall']
    precision = precision_score(testing_classes, testpredictions)

    l_sensitivity.append(sensitivity)
    l_specificity.append(specificity)
    l_precision.append(precision)


"""
Print average
"""
print ("logistic train auc is: ", np.mean(logistic_train_auc))
print("logistic test auc is: ", np.mean(logistic_test_auc))

print ("logistic train accuracy is: ", np.mean(logistic_train_accuracy))
print("logistic test accuracy is: ", np.mean(logistic_test_accuracy))

print("")

print("Logistic info:")
print("sensitivity: ", np.mean(l_sensitivity))
print("specificity: ", np.mean(l_specificity))
print("precision: ", np.mean(l_precision))
