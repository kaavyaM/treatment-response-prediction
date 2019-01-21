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

# Classification report

k_sensitivity = []
k_specificity = []
k_precision = []



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
    Logistic: Recursive Feature elimination
    """

    logisticRegr = LogisticRegression(max_iter=1000)
    rfe = RFE(logisticRegr, n_features_to_select=50, verbose=0)
    fit_logistic = rfe.fit(training_inputs, training_classes)

    """
    Feature selection (Change to fit_dtc when running RFE with DTC)
    """
    low_rankings = []
    for i in range(fit_logistic.support_.size):
        if fit_logistic.support_[i]:
            low_rankings.append(i)
    training_inputs = training_inputs[:, low_rankings]
    testing_inputs = testing_inputs[:, low_rankings]


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
Print average
"""

# KNN data

print ("knn train auc is: ", np.mean(knn_train_auc))
print("knn test auc is: ", np.mean(knn_test_auc))

print ("knn train accuracy is: ", np.mean(knn_train_accuracy))
print("knn test accuracy is: ", np.mean(knn_test_accuracy))



print("KNN info:")
print("sensitivity: ", np.mean(k_sensitivity))
print("specificity: ", np.mean(k_specificity))
print("precision: ", np.mean(k_precision))

