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

# Proportion to downscale positive samples by
downsampling_factor = 0.8

data = np.genfromtxt('student_data_minimized.csv.csv', delimiter=',', skip_header=1, dtype=float)

# For (i, J), map values for feature at index i to 1 if value is in J
# e.g., [1, 2, 5, 4, 5] is mapped to [1, 0, 1, 0, 1] for J=[1, 5]
binarize_features = [(7, [1]),
                        (8, [1]),
                        (9, [1]),
                        (10, [1]),
                        (11, [1]),
                        (12, [1]),
                        (13, [1]),
                        (14, [1]),
                        (15, [1,3])]

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

features = data[:,:-1]
classes = data[:-1]

disabilities = (7, 8, 9, 10, 11, 12, 13, 14)
# Combine disability categories
imputed_disability = []
for d in data:
    if 1 in d[7:14]:
        imputed_disability.append(1)
    else:
        imputed_disability.append(0)
imputed_disability = np.array(imputed_disability)

# Remove disability columns & replace with imputed version
features = np.delete(features, disabilities, axis=1)
features = np.append(features, imputed_disability.reshape((len(imputed_disability), -1)), axis=1)

print(features)