from data.DataLoader import DataLoader
from utils.Utils import Utils
from sklearn.mixture import GaussianMixture
import numpy as np
from joblib import Parallel, delayed
import pandas as pd 

dataLoader = DataLoader()
utils = Utils()

train_data, val_data, test_data = dataLoader.loadData()
val_labels, test_labels = dataLoader.loadLabels()

def train_gmm(n, q):
    gmm = GaussianMixture(n_components=n, max_iter=10 ** 5, tol=10 ** -5, random_state=42)
    gmm.fit(train_data)
    
    scores = gmm.score_samples(val_data)
    threshold = np.quantile(scores, q)
    
    idxAnomaly = np.where(scores <= threshold)
    idxNormal = np.where(scores > threshold)
    
    val_predictions = scores
    
    val_predictions[idxAnomaly] = 1
    val_predictions[idxNormal] = 0
    
    accuracy, recall, precision, f1 = utils.performanceMetrics(val_predictions, val_labels)
    
    return (n, q, accuracy, recall, precision, f1)

results_gmm = Parallel(n_jobs=-1)(
    delayed(train_gmm)(n, q) for n in range(1, 21) for q in [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
)       

df_gmm = pd.DataFrame(
    {"components": [],
    "quantile": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for (n, q, accuracy, recall, precision, f1) in results_gmm:
    df_gmm = pd.concat([df_gmm, pd.DataFrame( 
        {"components": [n],
        "quantile": [q],
        "accuracy": [accuracy],
        "recall" : [recall],
        "precision" : [precision],
        "f1" : [f1]},
    dtype=object
    )], 
        ignore_index=True)
    
    df_gmm.to_csv("gmm.csv", index=False)