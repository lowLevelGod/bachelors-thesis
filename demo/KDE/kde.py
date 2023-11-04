from data.DataLoader import DataLoader
from utils.Utils import Utils
from sklearn.neighbors import KernelDensity
import numpy as np
from joblib import Parallel, delayed
import pandas as pd 

dataLoader = DataLoader()
utils = Utils()

train_data, val_data, test_data = dataLoader.loadData()
val_labels, test_labels = dataLoader.loadLabels()

def train_kde(k, b, q):
    kde = KernelDensity(kernel=k, bandwidth=b)
    kde.fit(train_data)

    scores = kde.score_samples(val_data)
    threshold = np.quantile(scores, q)

    idxAnomaly = np.where(scores <= threshold)
    idxNormal = np.where(scores > threshold)

    val_predictions = scores

    val_predictions[idxAnomaly] = 1
    val_predictions[idxNormal] = 0

    accuracy, recall, precision, f1 = utils.performanceMetrics(val_predictions, val_labels)
    
    return (k, b, q, accuracy, recall, precision, f1)

results_kde = Parallel(n_jobs=-1)(
    delayed(train_kde)(k, b, q) for k in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'] for b in [0.5, 1.0, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0] for q in [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
)       

df_kde = pd.DataFrame(
    {"kernel": [],
    "bandwidth": [],
    "quantile": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for (k, b, q, accuracy, recall, precision, f1) in results_kde:
    df_kde = pd.concat([df_kde, pd.DataFrame( 
        {"kernel": [k],
        "bandwidth": [b],
        "quantile": [q],
        "accuracy": [accuracy],
        "recall" : [recall],
        "precision" : [precision],
        "f1" : [f1]},
    dtype=object
    )], 
        ignore_index=True)

    df_kde.to_csv("kde.csv", index=False)
