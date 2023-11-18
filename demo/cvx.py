import cvxpy as cp
import numpy as np 
from data.DataLoader import *
from cvx_model import *
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

dl = DataLoader()
train_data, val_data, test_data = dl.loadData()
val_labels, test_labels = dl.loadLabels()

n = int(0.001 * len(train_data))
gamma = 0.01
nu = 0.001

X = train_data[np.random.choice(len(train_data), size=n, replace=False)]

cvx = CvxModel(kernel='rbf', nu=nu, gamma=gamma)
cvx.fit(X)

val_predictions = cvx.predict(val_data)

conf_mat = confusion_matrix(val_labels, val_predictions)
tn = conf_mat[0][0]
tp = conf_mat[1][1]
fp = conf_mat[0][1]
fn = conf_mat[1][0]
accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
recall = recall_score(val_labels, val_predictions)
precision = precision_score(val_labels, val_predictions)
f1 = f1_score(val_labels, val_predictions)

print("accuracy= ", accuracy)
print("recall= ", recall)
print("precision= ", precision)
print("f1= ", f1)
