from data.DataLoader import DataLoader
from utils.Utils import Utils
from sklearn.svm import OneClassSVM
from joblib import Parallel, delayed
import pandas as pd 

dataLoader = DataLoader()
utils = Utils()

train_data, val_data, test_data = dataLoader.loadData()
val_labels, test_labels = dataLoader.loadLabels()

def train_ocsvm_rbf(g, n):
    ocsvm = OneClassSVM(kernel='rbf', gamma=g, nu=n)
    ocsvm.fit(train_data)
    val_predictions = ocsvm.predict(val_data)
    val_predictions[val_predictions == 1] = 0
    val_predictions[val_predictions == -1] = 1
    
    accuracy, recall, precision, f1 = utils.performanceMetrics(val_predictions, val_labels)
    
    return (g, n, accuracy, recall, precision, f1)
    
results_ocsvm_rbf = Parallel(n_jobs=-1)(
    delayed(train_ocsvm_rbf)(g, n) for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0] for n in [10 ** -4, 10 ** -3, 5 * 10 ** -3, 7 * 10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
)       

df_ocsvm_rbf = pd.DataFrame(
    {"gamma": [],
    "nu": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for (g, n, accuracy, recall, precision, f1) in results_ocsvm_rbf:
    df_ocsvm_rbf = pd.concat([df_ocsvm_rbf, pd.DataFrame( 
        {"gamma": [g],
        "nu": [n],
        "accuracy": [accuracy],
        "recall" : [recall],
        "precision" : [precision],
        "f1" : [f1]},
    dtype=object
    )], 
        ignore_index=True)

    df_ocsvm_rbf.to_csv("ocsvm-rbf.csv", index=False)