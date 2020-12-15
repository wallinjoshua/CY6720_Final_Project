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
from sklearn.tree import DecisionTreeClassifier
from CalibratedAdaMEC import CalibratedAdaMECClassifier # Our calibrated AdaMEC implementation can be found here
from sklearn.inspection import permutation_importance

feature_names = [
                'Student ID',
                'Sex',
                'Race/Ethnicity',
                'Is English first language?',
                'English self-efficacy score',
                'Student\'s native language',
                'Student\'s English fluency',
                'Socio-economic status version 1',
                'Socio-economic status version 2',
                '10th grade enrollment at student\'s school',
                'Urbanicity of school',
                'Region of school',
                'Tardiness Rate English',
                'Tardiness Rate Math',
                'In-school suspensions',
                'Suspended or put on probation',
                'How many grades repeated',
                'Last grade repeated by student',
                'Student has an IEP',
                'Student Math standardized test score',
                'Student Reading standardized test score',
                'Student PISA Math score',
                'Student PISA Reading score',
                'How far student thinks they\'ll get in their education',
                'How far parent wants student to go in their education',
                'How successful student expects to be in learning',
                'How much effort and persistence student perceives in self',
                'How many hours does student spend on homework per week',
                'Took or plans to take the PSAT',
                'Completed high school by second follow-up'
]

#binarize_features = [(18, [1]), (28, [3,4,5]), (29, [1,3])]
# clf = load('adaboost_best_trained_classifier.joblib')
clf = load('adaboost_best_minimized_permutation.joblib')
data = np.genfromtxt('converted_student_data_2009.csv', delimiter=',', skip_header=1, dtype=float)

# for c, h in binarize_features:
#     new_col = []
#     for v in data[:,c]:
#         if v in h:
#             new_col.append(1)
#         else:
#             new_col.append(0)
#     data[:,c] = np.array(new_col)
data = np.delete(data, 0, axis=1)

feature_test = data[:,:-1]
class_test = data[:,-1]

preds_clf = clf.predict(feature_test)
tn_clf, fp_clf, fn_clf, tp_clf = confusion_matrix(class_test, preds_clf).ravel()
print("\tAdaboost Metrics:")
print("\t\tOverall Accuracy:", accuracy_score(class_test, preds_clf))
print("\t\tNegative Class Accuracy:", tn_clf/(tn_clf+fp_clf))

recall = tn_clf/(tn_clf+fp_clf) #How many of the non-graduating students did I identify?
precision = tn_clf/(tn_clf+fn_clf) #How many of the people I said were non-graduating actually were?

print("\t\tRecall:", recall)
print("\t\tPrecision:", precision)
print("\t\tF-Measure:", (2 * recall * precision)/(recall + precision))
print("\t\tG-Mean:", math.sqrt((tp_clf/(tp_clf+fn_clf)) * (tn_clf/(tn_clf+fp_clf))))

# print("R^2", clf.score(feature_test, class_test))

# print("FEATURE IMPORTANCE WITHOUT PERMUTATION:")

# feature_importance_dictionary = list(zip(np.argsort(clf.feature_importances_), np.sort(clf.feature_importances_)))
# feature_importance_dictionary.reverse()
# feature_importance_dictionary = [(i+1, j) for (i, j) in feature_importance_dictionary]
# for i in feature_importance_dictionary:
#     print(feature_names[i[0]], ": ", i)

# print("\n")

# print("PERMUTATION FEATURE IMPORTANCE:")

# perm_importance = permutation_importance(clf, feature_test, class_test, n_repeats=10, random_state=0)

# perm_feature_importance_dictionary = list(zip(np.argsort(perm_importance.importances_mean), np.sort(perm_importance.importances_mean)))
# perm_feature_importance_dictionary.reverse()
# perm_feature_importance_dictionary = [(i+1, j) for (i, j) in perm_feature_importance_dictionary]
# for i in perm_feature_importance_dictionary:
#     print(feature_names[i[0]], ": ", i)

# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))

# pos = np.arange(len(feature_importance_dictionary))

# bars = ax1.barh(pos, [i[1] for i in feature_importance_dictionary],
#                     align='center',
#                     height=0.3,
#                     tick_label=[('Feature ' + str(i[0])) for i in feature_importance_dictionary])
# ax1.set_xlabel("Gini Importance")
# ax1.set_title("Impurity-Based Feature Importance")

# bars = ax2.barh(pos, [i[1] for i in perm_feature_importance_dictionary],
#                     align='center',
#                     height=0.3,
#                     tick_label=[('Feature ' + str(i[0])) for i in perm_feature_importance_dictionary])
# ax2.set_xlabel("Mean Increase in Error when Permuted")
# ax2.set_title("Permutation Feature Importance")

# plt.show()