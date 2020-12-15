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
from sklearn.utils import resample
import matplotlib.pyplot as plt
import math

categorical = (0, 1, 2, 3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24)

# Proportion to downscale positive samples by
downsampling_factor = 0.0

# For (i, J), map values for feature at index i to 1 if value is in J
# e.g., [1, 2, 5, 4, 5] is mapped to [1, 0, 1, 0, 1] for J=[1, 5]
binarize_features = [(18, [1]), (28, [3,4,5]), (29, [1,3])]

data = np.genfromtxt('student_data_v2.csv', delimiter=',', skip_header=1, dtype=float)

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
feature_train, feature_test, class_train, class_test =  train_test_split(feature_labels, class_labels, train_size = 0.8)

# # Upsample 20% of negative training samples
# combined_training_data = np.append(feature_train, class_train.reshape((len(class_train),-1)), axis=1)
# negative_samples = np.array([x for x in combined_training_data if x[28] == 0])
# new_samples = resample(negative_samples, n_samples=int(10 * len(negative_samples)))
# combined_training_data = np.append(combined_training_data, new_samples, axis=0)
# feature_train = combined_training_data[:, :-1]
# class_train = combined_training_data[:,-1]

knn_n_accuracies = []
svc_n_accuracies = []
rfc_n_accuracies = []
ada_n_accuracies = []

knn_all_accuracies = []
svc_all_accuracies = []
rfc_all_accuracies = []
ada_all_accuracies = []

max_downsampling = 0.95
downsampling_step = 0.05
combined_training_data = np.append(feature_train, class_train.reshape((len(class_train),-1)), axis=1)
positive_samples = np.array([x for x in combined_training_data if x[28] == 1])
negative_samples = np.array([x for x in combined_training_data if x[28] == 0])
for downsampling_factor in np.linspace(0, max_downsampling, num=int(math.ceil((max_downsampling-0)/downsampling_step)) + 1):

    print("REDUCING POSITIVE SAMPLE SIZE BY %2.2f%%" % (downsampling_factor*100))

    # Downsample the positive training examples
    new_samples = resample(positive_samples, n_samples=int(math.ceil((1-downsampling_factor) * len(positive_samples))))
    combined_training_data_2 = np.append(negative_samples, new_samples, axis=0)
    feature_train = combined_training_data_2[:, :-1]
    class_train = combined_training_data_2[:,-1]

    print("Proportion of Positive to Negative samples: %2.2f:1" % ((len(positive_samples)*(1-downsampling_factor))/len(negative_samples)))

    # Create pipelines to scale and train
    pipeline_knn = make_pipeline(
        KNeighborsClassifier(n_neighbors=3)
    )

    pipeline_svc = make_pipeline(
        SVC()
    )

    pipeline_rfc = make_pipeline(
        RandomForestClassifier(n_estimators=100)
    )

    clf = AdaBoostClassifier(n_estimators=1000)
    clf.fit(feature_train, class_train)

    pipeline_knn.fit(feature_train, class_train)
    pipeline_svc.fit(feature_train, class_train)
    pipeline_rfc.fit(feature_train, class_train)
    clf.fit(feature_train, class_train)

    print("*5-FOLD ACCURACIES*")

    scores_knn = cross_val_score(pipeline_knn, feature_train, class_train, cv=5)
    print("5-Fold Accuracy for KNN: %0.2f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std() * 2))

    scores_svc = cross_val_score(pipeline_svc, feature_train, class_train, cv=5)
    print("5-fold Accuracy for SVC: %0.2f (+/- %0.2f)" % (scores_svc.mean(), scores_svc.std() * 2))

    scores_rfc = cross_val_score(pipeline_rfc, feature_train, class_train, cv=5)
    print("5-fold Accuracy for Random Forest: %0.2f (+/- %0.2f)" % (scores_rfc.mean(), scores_rfc.std() * 2))

    scores_clf = cross_val_score(clf, feature_train, class_train, cv=5)
    print("5-fold Accuracy for Adaboost: %0.2f (+/- %0.2f)" % (scores_clf.mean(), scores_clf.std() * 2))

    preds_knn = pipeline_knn.predict(feature_test)
    preds_svc = pipeline_svc.predict(feature_test)
    preds_rfc = pipeline_rfc.predict(feature_test)
    preds_clf = clf.predict(feature_test)

    tn_knn, fp_knn, fn_knn, tp_knn = confusion_matrix(class_test, preds_knn).ravel()
    tn_svc, fp_svc, fn_svc, tp_svc = confusion_matrix(class_test, preds_svc).ravel()
    tn_rfc, fp_rfc, fn_rfc, tp_rfc = confusion_matrix(class_test, preds_rfc).ravel()
    tn_clf, fp_clf, fn_clf, tp_clf = confusion_matrix(class_test, preds_clf).ravel()

    knn_n_accuracies.append(tn_knn/(tn_knn+fp_knn))
    svc_n_accuracies.append(tn_svc/(tn_svc+fp_svc))
    rfc_n_accuracies.append(tn_rfc/(tn_rfc+fp_rfc))
    ada_n_accuracies.append(tn_clf/(tn_clf+fp_clf))

    knn_all_accuracies.append(accuracy_score(class_test, preds_knn))
    svc_all_accuracies.append(accuracy_score(class_test, preds_svc))
    rfc_all_accuracies.append(accuracy_score(class_test, preds_rfc))
    ada_all_accuracies.append(accuracy_score(class_test, preds_clf))

    print("KNN Accuracy:")
    print("\t\tOverall:", accuracy_score(class_test, preds_knn))
    print("\t\tNegative Class:", tn_knn/(tn_knn+fp_knn))

    print("SVC Accuracy:")
    print("\t\tOverall:", accuracy_score(class_test, preds_svc))
    print("\t\tNegative Class:", tn_svc/(tn_svc+fp_svc))

    print("Random Forest Accuracy:")
    print("\t\tOverall:", accuracy_score(class_test, preds_rfc))
    print("\t\tNegative Class:", tn_rfc/(tn_rfc+fp_rfc))

    print("Adaboost Accuracy:")
    print("\t\tOverall:", accuracy_score(class_test, preds_clf))
    print("\t\tNegative Class:", tn_clf/(tn_clf+fp_clf))

knn_n_accuracies = np.array(knn_n_accuracies)
svc_n_accuracies = np.array(svc_n_accuracies)
rfc_n_accuracies = np.array(rfc_n_accuracies)
ada_n_accuracies = np.array(ada_n_accuracies)

knn_all_accuracies = np.array(knn_all_accuracies)
svc_all_accuracies = np.array(svc_all_accuracies)
rfc_all_accuracies = np.array(rfc_all_accuracies)
ada_all_accuracies = np.array(ada_all_accuracies)

x = np.linspace(0, max_downsampling, num=int(math.ceil((max_downsampling-0)/downsampling_step)) + 1)

# fig, (plt, plt) = plt.subplots(1, 2)
plt.figure(figsize=(12, 6))

svm_n = plt.plot(x, svc_n_accuracies, color='r', label='SVM Negative')
knn_n = plt.plot(x, knn_n_accuracies, color='g', label='KNN (k=3) Negative')
rfc_n = plt.plot(x, rfc_n_accuracies, color='b', label='Random Forest Negative')
ada_n = plt.plot(x, ada_n_accuracies, color='m', label='Adaboost Negative')

svm_all = plt.plot(x, svc_all_accuracies, color='r', ls='--', label='SVM Overall')
knn_all = plt.plot(x, knn_all_accuracies, color='g', ls='--', label='KNN (k=3) Overall')
rfc_all = plt.plot(x, rfc_all_accuracies, color='b', ls='--', label='Random Forest Overall')
ada_all = plt.plot(x, ada_all_accuracies, color='m', ls='--', label='Adaboost Overall')

# svm_intersect = np.argwhere(np.diff(np.sign(svc_n_accuracies - svc_all_accuracies))).flatten()[0]
# knn_intersect = np.argwhere(np.diff(np.sign(knn_n_accuracies - knn_all_accuracies))).flatten()[0][0]
# rfc_intersect = np.argwhere(np.diff(np.sign(rfc_n_accuracies- rfc_all_accuracies))).flatten()[0][0]
# ada_intersect = np.argwhere(np.diff(np.sign(ada_n_accuracies - ada_all_accuracies))).flatten()[0][0]

# plt.axvline(x=svm_intersect, color='r', linestyle='-')
# plt.axvline(x=knn_intersect, color='g', linestyle='-')
# plt.axvline(x=rfc_intersect, color='b', linestyle='-')
# plt.axvline(x=ada_intersect, color='m', linestyle='-')

plt.axis([0, max_downsampling, 0, 1])
plt.xlabel("Downsampling Factor")
plt.ylabel("Accuracy")
plt.xticks(np.linspace(0, 1.0, num=11))
plt.title("Accuracy as Downsampling Increases")

plt.legend(loc="lower right", ncol=1)

plt.show()

# print(np.sort(pipeline_clf[0].feature_importances_))
# print([x + 1 for x in np.argsort(pipeline_clf[0].feature_importances_)])