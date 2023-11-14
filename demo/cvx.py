import cvxpy as cp
import numpy as np 
from data.DataLoader import *
from cvx_model import *

dl = DataLoader()
train_data, val_data, test_data = dl.loadData()
val_labels, test_labels = dl.loadLabels()

n = int(0.001 * len(train_data))
gamma = 2.0
nu = 0.1

X = train_data[np.random.choice(len(train_data), size=n, replace=False)]

cvx = CvxModel(kernel='linear', nu=nu, gamma=gamma)
cvx.fit(X)

val_predictions = cvx.predict(val_data[:1])

print(val_predictions)
