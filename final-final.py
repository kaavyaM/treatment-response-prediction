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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


from sklearn.neighbors import KNeighborsClassifier

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

knn_test_accuracy = []
knn_train_accuracy = []
knn_test_auc = []
knn_train_auc = []

dtc_test_accuracy = []
dtc_train_accuracy = []
dtc_test_auc = []
dtc_train_auc = []

logistic_test_accuracy = []
logistic_train_accuracy = []
logistic_test_auc = []
logistic_train_auc = []

mfnn_test_accuracy = []
mfnn_train_accuracy = []
mfnn_test_auc = []
mfnn_train_auc = []

svc_test_accuracy = []
svc_train_accuracy = []
svc_test_auc = []
svc_train_auc = []

# Classification report

d_sensitivity = []
d_specificity = []
d_precision = []

k_sensitivity = []
k_specificity = []
k_precision = []

l_sensitivity = []
l_specificity = []
l_precision = []

m_sensitivity = []
m_specificity = []
m_precision = []

s_sensitivity = []
s_specificity = []
s_precision = []

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
    Logistic: Recursive Feature elimination (No Hyperparameters)
    """

    logisticRegr = LogisticRegression(max_iter=1000)
    rfe = RFE(logisticRegr, n_features_to_select=50, verbose=0)
    fit_logistic = rfe.fit(training_inputs, training_classes)
    # print("Num Features: %d") % fit.n_features_
    # print("Selected Features: %s") % fit.support_
    # print("Feature Ranking: %s") % fit.ranking_

    """
    DTC: Recursive Feature Elimination
    """
    dtc = DecisionTreeClassifier(max_depth=1, min_samples_leaf= 0.2, random_state=42)
    rfe = RFE(dtc, n_features_to_select=10, verbose=0)
    fit_dtc = rfe.fit(training_inputs, training_classes)

    """
    Feature selection (Change to fit_dtc when running RFE with DTC)
    """
    low_rankings = []
    for i in range(fit_logistic.support_.size):
        if fit_logistic.support_[i]:
            low_rankings.append(i)
            # if data.columns[i] in best_features:
            #     best_features[data.columns[i]] += 1
            # else:
            #     best_features[data.columns[i]] = 1
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
    K Nearest Neighbors
    """

    # Fits classifier
    knn = KNeighborsClassifier(n_neighbors=25, p=3)
    knn = knn.fit(training_inputs, training_classes)

    # Predictions
    trainpredictions = knn.predict(training_inputs)
    testpredictions = knn.predict(testing_inputs)

    # Appending data for train and test data
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
    Logistic
    """

    # Fits classifier
    logisticRegr.fit(training_inputs, training_classes)

    # Predictions
    trainpredictions = logisticRegr.predict(training_inputs)
    testpredictions = logisticRegr.predict(testing_inputs)

    # Appending data for train and test data
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
    Multi feed neural network (Slow)
    """

    # Fits classifier
    mlp = MLPClassifier(random_state=42, max_iter=1000)
    mlp.fit(training_inputs, training_classes)

    # Predictions
    testpredictions = mlp.predict(testing_inputs)
    trainpredictions = mlp.predict(training_inputs)

    # Appending data for train and test data
    mfnn_test_accuracy.append(metrics.accuracy_score(testing_classes, testpredictions))
    mfnn_train_accuracy.append(metrics.accuracy_score(training_classes, trainpredictions))

    fpr, tpr, _ = metrics.roc_curve(testing_classes, testpredictions)
    mfnn_test_auc.append(metrics.auc(fpr, tpr))
    fpr2, tpr2, _ = metrics.roc_curve(training_classes, trainpredictions)
    mfnn_train_auc.append(metrics.auc(fpr2, tpr2))

    report = classification_report(testing_classes, testpredictions, output_dict=True)
    sensitivity = report['1']['recall']
    specificity = report['0']['recall']
    precision = precision_score(testing_classes, testpredictions)

    m_sensitivity.append(sensitivity)
    m_specificity.append(specificity)
    m_precision.append(precision)

    """
    SVM: No hyperparameters
    """
    # Fits classifier
    svc = SVC(random_state=42)
    svc.fit(training_inputs, training_classes)

    # Predictions
    testpredictions = svc.predict(testing_inputs)
    trainpredictions = svc.predict(training_inputs)

    # Appending data for train and test data
    svc_test_accuracy.append(metrics.accuracy_score(testing_classes, testpredictions))
    svc_train_accuracy.append(metrics.accuracy_score(training_classes, trainpredictions))

    fpr, tpr, _ = metrics.roc_curve(testing_classes, testpredictions)
    svc_test_auc.append(metrics.auc(fpr, tpr))
    fpr2, tpr2, _ = metrics.roc_curve(training_classes, trainpredictions)
    svc_train_auc.append(metrics.auc(fpr2, tpr2))

    report = classification_report(testing_classes, testpredictions, output_dict=True)
    sensitivity = report['1']['recall']
    specificity = report['0']['recall']
    precision = precision_score(testing_classes, testpredictions)

    s_sensitivity.append(sensitivity)
    s_specificity.append(specificity)
    s_precision.append(precision)

"""
Print average
"""
# DecisionTree data:

print ("dtc train auc is: ", np.mean(dtc_train_auc))
print("dtc test auc is: ", np.mean(dtc_test_auc))

print ("dtc train accuracy is: ", np.mean(dtc_train_accuracy))
print("dtc test accuracy is: ", np.mean(dtc_test_accuracy))

# KNN data

print ("knn train auc is: ", np.mean(knn_train_auc))
print("knn test auc is: ", np.mean(knn_test_auc))

print ("knn train accuracy is: ", np.mean(knn_train_accuracy))
print("knn test accuracy is: ", np.mean(knn_test_accuracy))

# Logistic data

print ("logistic train auc is: ", np.mean(logistic_train_auc))
print("logistic test auc is: ", np.mean(logistic_test_auc))

print ("logistic train accuracy is: ", np.mean(logistic_train_accuracy))
print("logistic test accuracy is: ", np.mean(logistic_test_accuracy))

# MFNN data

print ("mfnn train auc is: ", np.mean(mfnn_train_auc))
print("mfnn test auc is: ", np.mean(mfnn_test_auc))

print ("mfnn train accuracy is: ", np.mean(mfnn_train_accuracy))
print("mfnn test accuracy is: ", np.mean(mfnn_test_accuracy))

# More data

print("")

print("DTC info:")
print("sensitivity: ", np.mean(d_sensitivity))
print("specificity: ", np.mean(d_specificity))
print("precision: ", np.mean(d_precision))

print("")

print("KNN info:")
print("sensitivity: ", np.mean(k_sensitivity))
print("specificity: ", np.mean(k_specificity))
print("precision: ", np.mean(k_precision))

print("")

print("Logistic info:")
print("sensitivity: ", np.mean(l_sensitivity))
print("specificity: ", np.mean(l_specificity))
print("precision: ", np.mean(l_precision))

print("")

print("MFNN info:")
print("sensitivity: ", np.mean(m_sensitivity))
print("specificity: ", np.mean(m_specificity))
print("precision: ", np.mean(m_precision))

print("")

print("SVM info:")
print("sensitivity: ", np.mean(s_sensitivity))
print("specificity: ", np.mean(s_specificity))
print("precision: ", np.mean(s_precision))

