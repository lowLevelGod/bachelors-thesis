from data.DataLoader import DataLoader
from utils.Utils import Utils
from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem
from joblib import Parallel, delayed
import pandas as pd 
from sklearn import pipeline

dataLoader = DataLoader()
utils = Utils()

train_data, val_data, test_data = dataLoader.loadData()
val_labels, test_labels = dataLoader.loadLabels()

def train_nystroem_sigmoid(g, n):
    feature_map_nystroem = Nystroem(kernel='sigmoid',
                                    gamma=g,
                                random_state=42,
                            n_components=int(n * len(train_data)))

    nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                            ("svm", linear_model.SGDOneClassSVM(random_state=42))])


    nystroem_approx_svm.fit(train_data)

    val_predictions = nystroem_approx_svm.predict(val_data)

    val_predictions[val_predictions == 1] = 0
    val_predictions[val_predictions == -1] = 1
    
    accuracy, recall, precision, f1 = utils.performanceMetrics(val_predictions, val_labels)
        
    return (g, n, accuracy, recall, precision, f1)

results_nystroem_sigmoid = Parallel(n_jobs=-1)(
    delayed(train_nystroem_sigmoid)(g, n) for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0] for n in [10 ** -4, 10 ** -3, 10 ** -2, 4 * 10 ** -2, 5 * 10 ** -2, 10 ** -1, 3 * 10 ** -1]
)       

df_nystroem_sigmoid = pd.DataFrame(
    {"gamma": [],
    "components_percent": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for (g, n, accuracy, recall, precision, f1) in results_nystroem_sigmoid:
    df_nystroem_sigmoid = pd.concat([df_nystroem_sigmoid, pd.DataFrame( 
        {"gamma": [g],
        "components_percent": [n],
        "accuracy": [accuracy],
        "recall" : [recall],
        "precision" : [precision],
        "f1" : [f1]},
    dtype=object
    )], 
        ignore_index=True)
    
    df_nystroem_sigmoid.to_csv("nystroem-sigmoid.csv", index=False)
    


