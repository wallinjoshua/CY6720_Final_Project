import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample
import matplotlib.pyplot as plt
import math
from joblib import dump, load

# Proportion to downscale positive samples by
downsampling_factor = 0.8

# MODEL MINIMIZATION VIA IMPURITY-BASED FEATURE IMPORTANCE

# For (i, J), map values for feature at index i to 1 if value is in J
# e.g., [1, 2, 5, 4, 5] is mapped to [1, 0, 1, 0, 1] for J=[1, 5]
binarize_features = [(11, [1,3])]

data = np.genfromtxt('student_data_minimized_impurity.csv', delimiter=',', skip_header=1, dtype=float)

# BINARIZE RELEVANT FEATURES 
# (e.g., convert graduation timings, missing values, etc. to "Graduated" or "Did not graduate")
for c, h in binarize_features:
    new_col = []
    for v in data[:,c]:
        if v in h:
            new_col.append(1)
        else:
            new_col.append(0)
    data[:,c] = np.array(new_col)

# Remove the Student IDs row (not used in fitting)
data = np.delete(data, 0, axis=1)

feature_labels = data[:, :-1]
class_labels = data[:,-1]

best_overall_accuracy = 0.0
best_negative_accuracy = 0.0
best_clf_minimized_impurity = 0
for k in range(50):
    print("Test", k+1)
    feature_train, feature_test, class_train, class_test =  train_test_split(feature_labels, class_labels, train_size = 0.8)

    # Downsample the positive training examples
    combined_training_data = np.append(feature_train, class_train.reshape((len(class_train),-1)), axis=1)
    positive_samples = np.array([x for x in combined_training_data if x[10] == 1])
    negative_samples = np.array([x for x in combined_training_data if x[10] == 0])
    new_samples = resample(positive_samples, n_samples=int(math.ceil((1-downsampling_factor) * len(positive_samples))))
    combined_training_data = np.append(negative_samples, new_samples, axis=0)
    feature_train = combined_training_data[:, :-1]
    class_train = combined_training_data[:,-1]
    clf = AdaBoostClassifier(n_estimators=1000)

    clf.fit(feature_train, class_train)
    preds_clf = clf.predict(feature_test)
    tn_clf, fp_clf, fn_clf, tp_clf = confusion_matrix(class_test, preds_clf).ravel()
    print("\tAdaboost Accuracy:")
    print("\t\tOverall:", accuracy_score(class_test, preds_clf))
    print("\t\tNegative Class:", tn_clf/(tn_clf+fp_clf))

    recall = tn_clf/(tn_clf+fp_clf)
    precision = tn_clf/(tn_clf+fn_clf)

    print("\t\tRecall:", recall)
    print("\t\tPrecision:", precision)
    print("\t\tF-Measure:", (2 * recall * precision)/(recall + precision))
    print("\t\tG-Mean:", math.sqrt((tp_clf/(tp_clf+fn_clf)) * (tn_clf/(tn_clf+fp_clf))))

    if(accuracy_score(class_test, preds_clf) > best_overall_accuracy and tn_clf/(tn_clf+fp_clf) > best_negative_accuracy):
        best_overall_accuracy = accuracy_score(class_test, preds_clf)
        best_negative_accuracy = tn_clf/(tn_clf+fp_clf)
        best_clf_minimized_impurity = clf
    print("\t\tBest Overall Accuracy:", best_overall_accuracy)
    print("\t\tBest Negative Accuracy:", best_negative_accuracy)


data = np.genfromtxt('student_data_minimized_permutation.csv', delimiter=',', skip_header=1, dtype=float)

# BINARIZE RELEVANT FEATURES 
# (e.g., convert graduation timings, missing values, etc. to "Graduated" or "Did not graduate")
for c, h in binarize_features:
    new_col = []
    for v in data[:,c]:
        if v in h:
            new_col.append(1)
        else:
            new_col.append(0)
    data[:,c] = np.array(new_col)

# Remove the Student IDs row (not used in fitting)
data = np.delete(data, 0, axis=1)

feature_labels = data[:, :-1]
class_labels = data[:,-1]

best_overall_accuracy = 0.0
best_negative_accuracy = 0.0
best_clf_minimized_permutation = 0
for k in range(50):
    print("Test", k+1)
    feature_train, feature_test, class_train, class_test =  train_test_split(feature_labels, class_labels, train_size = 0.8)

    # Downsample the positive training examples
    combined_training_data = np.append(feature_train, class_train.reshape((len(class_train),-1)), axis=1)
    positive_samples = np.array([x for x in combined_training_data if x[10] == 1])
    negative_samples = np.array([x for x in combined_training_data if x[10] == 0])
    new_samples = resample(positive_samples, n_samples=int(math.ceil((1-downsampling_factor) * len(positive_samples))))
    combined_training_data = np.append(negative_samples, new_samples, axis=0)
    feature_train = combined_training_data[:, :-1]
    class_train = combined_training_data[:,-1]
    clf = AdaBoostClassifier(n_estimators=1000)

    clf.fit(feature_train, class_train)
    preds_clf = clf.predict(feature_test)
    tn_clf, fp_clf, fn_clf, tp_clf = confusion_matrix(class_test, preds_clf).ravel()
    print("\tAdaboost Accuracy:")
    print("\t\tOverall:", accuracy_score(class_test, preds_clf))
    print("\t\tNegative Class:", tn_clf/(tn_clf+fp_clf))

    recall = tn_clf/(tn_clf+fp_clf)
    precision = tn_clf/(tn_clf+fn_clf)

    print("\t\tRecall:", recall)
    print("\t\tPrecision:", precision)
    print("\t\tF-Measure:", (2 * recall * precision)/(recall + precision))
    print("\t\tG-Mean:", math.sqrt((tp_clf/(tp_clf+fn_clf)) * (tn_clf/(tn_clf+fp_clf))))

    if(accuracy_score(class_test, preds_clf) > best_overall_accuracy and tn_clf/(tn_clf+fp_clf) > best_negative_accuracy):
        best_overall_accuracy = accuracy_score(class_test, preds_clf)
        best_negative_accuracy = tn_clf/(tn_clf+fp_clf)
        best_clf_minimized_permutation = clf
    print("\t\tBest Overall Accuracy:", best_overall_accuracy)
    print("\t\tBest Negative Accuracy:", best_negative_accuracy)

dump(best_clf_minimized_impurity, 'adaboost_best_minimized_impurity.joblib')
dump(best_clf_minimized_permutation, 'adaboost_best_minimized_permutation.joblib')