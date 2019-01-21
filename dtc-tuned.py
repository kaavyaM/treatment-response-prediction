## Python version

"""
Most of these packages that we haven't used yet but might want to use in the future.
"""

import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, precision_score

"""
Some intro Data reading and description
"""
data = pd.read_csv('preprocessed.csv')

"""
Print data descriptors
"""
print(data.shape)
print (data)
print(data.describe())

# print how many null values are in each column
print(data.isnull().sum())

"""
Identifies all inputs and all outputs
"""
all_inputs = (data.drop(['Remission_Type'], axis=1)).values
all_labels = data['Remission_Type'].values


"""
Initialize arrays for Train and Test results
"""

dtc_test_accuracy = []
dtc_train_accuracy = []
dtc_test_auc = []
dtc_train_auc = []

# Classification report

d_sensitivity = []
d_specificity = []
d_precision = []


"""
K-fold validation
"""
from sklearn.model_selection import KFold
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
    dtc = DecisionTreeClassifier(max_depth=1, min_samples_leaf= 0.2, random_state=42)
    rfe = RFE(dtc, n_features_to_select=50, verbose=0)
    fit_dtc = rfe.fit(training_inputs, training_classes)

    """
    Feature selection (Change to fit_dtc when running RFE with DTC)
    """
    low_rankings = []
    for i in range(fit_dtc.support_.size):
        if fit_dtc.support_[i]:
            low_rankings.append(i)
    training_inputs = training_inputs[:, low_rankings]
    testing_inputs = testing_inputs[:, low_rankings]

    """
    Decision Tree classifier
    """
    dtc.fit(training_inputs, training_classes)

    # Predictions
    trainpredictions = dtc.predict(training_inputs)
    testpredictions = dtc.predict(testing_inputs)

    # Appending data for train and test data
    dtc_test_accuracy.append(metrics.accuracy_score(testing_classes, testpredictions))
    dtc_train_accuracy.append(metrics.accuracy_score(training_classes, trainpredictions))

    fpr, tpr, _ = metrics.roc_curve(testing_classes, testpredictions)
    dtc_test_auc.append(metrics.auc(fpr, tpr))
    fpr2, tpr2, _ = metrics.roc_curve(training_classes, trainpredictions)
    dtc_train_auc.append(metrics.auc(fpr2, tpr2))

    report = classification_report(testing_classes, testpredictions, output_dict=True)
    sensitivity = report['1']['recall']
    specificity = report['0']['recall']
    precision = precision_score(testing_classes, testpredictions)

    d_sensitivity.append(sensitivity)
    d_specificity.append(specificity)
    d_precision.append(precision)


"""
Print average
"""
# DecisionTree data:

print ("dtc train auc is: ", np.mean(dtc_train_auc))
print("dtc test auc is: ", np.mean(dtc_test_auc))

print ("dtc train accuracy is: ", np.mean(dtc_train_accuracy))
print("dtc test accuracy is: ", np.mean(dtc_test_accuracy))

# More data

print("")

print("DTC info:")
print("sensitivity: ", np.mean(d_sensitivity))
print("specificity: ", np.mean(d_specificity))
print("precision: ", np.mean(d_precision))

