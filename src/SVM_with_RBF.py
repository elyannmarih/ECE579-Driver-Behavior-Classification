import ast

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
path = "H:/CIS 585/FinalProject/ECE579-Driver-Behavior-Classification/data/train"


df = pd.read_csv(path+"/"+"D1traindataAveragemagnitudes50 (1).csv")
headers = list(df.columns)
print(headers)
X_train = df.iloc[:, 0].apply(ast.literal_eval).tolist()
X_train = np.array(X_train)
y_train = df.iloc[:, 1]
y_train = np.array(y_train)


def rbf_kernel(x,y,gamma=1.0):
      distance = np.linalg.norm(x - y) ** 2
      return np.exp(-gamma * distance)

svm_rbf = SVC(kernel="rbf", gamma=0.5, C=1.0)
print(X_train[0])
print(y_train[0])

svm_rbf.fit(X_train, y_train) 