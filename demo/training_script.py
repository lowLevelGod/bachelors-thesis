# %%
from pathlib import Path
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.kernel_approximation import Nystroem
from sklearn import pipeline
from joblib import Parallel, delayed
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import time

# %%
data = pd.read_csv(Path("./data/creditcard.csv"))
dataWithoutTime = data.drop("Time", axis = 1)

# %%
train_ratio = 0.75
validation_ratio = 0.25

splitterVal = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
splitterTest = StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=42)

# %%
train_index, val_index = list(splitterTest.split(dataWithoutTime, dataWithoutTime['Class']))[0]

train_data, val_data = dataWithoutTime.iloc[train_index], dataWithoutTime.iloc[val_index]

normal_train_data = train_data[train_data.Class == 0]
outlier_train_data = train_data[train_data.Class == 1]

val_data = pd.concat([val_data, outlier_train_data], ignore_index=True)

test_index, val_index = list(splitterVal.split(val_data, val_data['Class']))[0]

test_data, val_data = val_data.iloc[test_index], val_data.iloc[val_index]

# %%

test_labels = test_data['Class'].to_numpy()
val_labels = val_data['Class'].to_numpy()


# %%
train_data = normal_train_data.drop("Class", axis=1)
test_data = test_data.drop("Class", axis=1)
val_data = val_data.drop("Class", axis=1)

train_data = train_data.to_numpy()
test_data = test_data.to_numpy()
val_data = val_data.to_numpy()

# %%
scaler = StandardScaler()

train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

# %%

df_ocsvm_rbf = pd.DataFrame(
    {"gamma": [],
    "nu": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : [],
    "time" : []},
    dtype=object
)

def train_ocsvm_rbf(g, n):
    
    ocsvm = OneClassSVM(kernel='rbf', gamma=g, nu=n)
    
    start = time.time()
    ocsvm.fit(train_data)
    end = time.time()
        
    val_predictions = ocsvm.predict(val_data)
    val_predictions[val_predictions == 1] = 0
    val_predictions[val_predictions == -1] = 1
    conf_mat = confusion_matrix(val_labels, val_predictions)
    tn = conf_mat[0][0]
    tp = conf_mat[1][1]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    recall = recall_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)
    f1 = f1_score(val_labels, val_predictions)
    
    global df_ocsvm_rbf
    df_ocsvm_rbf = pd.concat([df_ocsvm_rbf, pd.DataFrame( 
            {"gamma": [g],
            "nu": [n],
            "accuracy": [accuracy],
            "recall" : [recall],
            "precision" : [precision],
            "f1" : [f1],
            "time" : [end - start]},
        dtype=object
        )], 
            ignore_index=True)

    df_ocsvm_rbf.to_csv("ocsvm-rbf.csv", index=False)
    
    return (g, n, accuracy, recall, precision, f1)
    
    
def process_ocsvm_rbf():
    results_ocsvm_rbf = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(train_ocsvm_rbf)(g, n) for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0] for n in [10 ** -4, 10 ** -3, 5 * 10 ** -3, 7 * 10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    )       
# %%

df_ocsvm_poly = pd.DataFrame(
    {"degree": [],
    "gamma": [],
    "nu": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : [],
    "time" : []},
    dtype=object
)

def train_ocsvm_poly(d, g, n):
    
    ocsvm = OneClassSVM(kernel='poly', degree=d, gamma=g, nu=n)
    
    start = time.time()
    ocsvm.fit(train_data)
    end = time.time()
        
    val_predictions = ocsvm.predict(val_data)
    val_predictions[val_predictions == 1] = 0
    val_predictions[val_predictions == -1] = 1
    conf_mat = confusion_matrix(val_labels, val_predictions)
    tn = conf_mat[0][0]
    tp = conf_mat[1][1]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    recall = recall_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)
    f1 = f1_score(val_labels, val_predictions)
    
    global df_ocsvm_poly
    df_ocsvm_poly = pd.concat([df_ocsvm_poly, pd.DataFrame( 
                {"degree": [d],
                "gamma": [g],
                "nu": [n],
                "accuracy": [accuracy],
                "recall" : [recall],
                "precision" : [precision],
                "f1" : [f1],
                "time" : [end - start]},
            dtype=object
            )], 
                ignore_index=True)

    df_ocsvm_poly.to_csv("ocsvm-poly.csv", index=False)
    
    return (d, g, n, accuracy, recall, precision, f1)
  
def process_ocsvm_poly():     
    results_ocsvm_poly = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(train_ocsvm_poly)(d, g, n) for d in range(1, 12, 1) for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0] for n in [10 ** -4, 10 ** -3, 5 * 10 ** -3, 7 * 10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    )       

# %%
df_ocsvm_sigmoid = pd.DataFrame(
    {"gamma": [],
    "nu": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : [],
    "time" : []},
    dtype=object
)


def train_ocsvm_sigmoid(g, n):
    ocsvm = OneClassSVM(kernel='sigmoid',gamma=g, nu=n)
    
    start = time.time()
    ocsvm.fit(train_data)
    end = time.time()
    
    val_predictions = ocsvm.predict(val_data)
    val_predictions[val_predictions == 1] = 0
    val_predictions[val_predictions == -1] = 1
    conf_mat = confusion_matrix(val_labels, val_predictions)
    tn = conf_mat[0][0]
    tp = conf_mat[1][1]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    recall = recall_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)
    f1 = f1_score(val_labels, val_predictions)
    
    global df_ocsvm_sigmoid
    df_ocsvm_sigmoid = pd.concat([df_ocsvm_sigmoid, pd.DataFrame( 
        {"gamma": [g],
        "nu": [n],
        "accuracy": [accuracy],
        "recall" : [recall],
        "precision" : [precision],
        "f1" : [f1],
        "time" : [end - start]},
    dtype=object
    )], 
        ignore_index=True)

    df_ocsvm_sigmoid.to_csv("ocsvm-sigmoid.csv", index=False)
    
    return (g, n, accuracy, recall, precision, f1)

def process_ocsvm_sigmoid():
    
    results_ocsvm_sigmoid = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(train_ocsvm_sigmoid)(g, n) for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0] for n in [10 ** -4, 10 ** -3, 5 * 10 ** -3, 7 * 10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    )     


# %%
df_ocsvm_linear = pd.DataFrame(
    {
    "nu": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : [],
    "time" : []},
    dtype=object
)


def train_ocsvm_linear(n):
    ocsvm = OneClassSVM(kernel='linear',  nu=n)
    
    start = time.time()
    ocsvm.fit(train_data)
    end = time.time()
    
    val_predictions = ocsvm.predict(val_data)
    val_predictions[val_predictions == 1] = 0
    val_predictions[val_predictions == -1] = 1
    conf_mat = confusion_matrix(val_labels, val_predictions)
    tn = conf_mat[0][0]
    tp = conf_mat[1][1]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    recall = recall_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)
    f1 = f1_score(val_labels, val_predictions)
    
    global df_ocsvm_linear
    df_ocsvm_linear = pd.concat([df_ocsvm_linear, pd.DataFrame( 
            {
            "nu": [n],
            "accuracy": [accuracy],
            "recall" : [recall],
            "precision" : [precision],
            "f1" : [f1],
            "time"  : [end - start]},
        dtype=object
        )], 
            ignore_index=True)

    df_ocsvm_linear.to_csv("ocsvm-linear.csv", index=False)
    
    return (n, accuracy, recall, precision, f1)
    
def process_ocsvm_linear():    
    results_ocsvm_linear = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(train_ocsvm_linear)(n) for n in [10 ** -4, 10 ** -3, 5 * 10 ** -3, 7 * 10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]
    )       

# %%
from sklearn.mixture import GaussianMixture

df_gmm = pd.DataFrame(
    {"components": [],
    "quantile": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : [],
    "time"  : []},
    dtype=object
)

def train_gmm(n, q):
    
    gmm = GaussianMixture(n_components=n, max_iter=10 ** 5, tol=10 ** -5, random_state=42)
    
    start = time.time()
    gmm.fit(train_data)
    end = time.time()
    
    
    scores = gmm.score_samples(val_data)
    threshold = np.quantile(scores, q)
    
    idxAnomaly = np.where(scores <= threshold)
    idxNormal = np.where(scores > threshold)
    
    val_predictions = scores
    
    val_predictions[idxAnomaly] = 1
    val_predictions[idxNormal] = 0
    
    conf_mat = confusion_matrix(val_labels, val_predictions)
    tn = conf_mat[0][0]
    tp = conf_mat[1][1]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    recall = recall_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)
    f1 = f1_score(val_labels, val_predictions)
    
    global df_gmm
    df_gmm = pd.concat([df_gmm, pd.DataFrame( 
            {"components": [n],
            "quantile": [q],
            "accuracy": [accuracy],
            "recall" : [recall],
            "precision" : [precision],
            "f1" : [f1],
            "time" : [end - start]},
        dtype=object
        )], 
            ignore_index=True)
        
    df_gmm.to_csv("gmm.csv", index=False)
    
    return (n, q, accuracy, recall, precision, f1)

def process_gmm():
    results_gmm = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(train_gmm)(n, q) for n in range(1, 21) for q in [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    )      

# %%
from sklearn.neighbors import KernelDensity

df_kde = pd.DataFrame(
    {"kernel": [],
    "bandwidth": [],
    "quantile": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : [],
    "time" : []},
    dtype=object
)

def train_kde(k, b, q):
    
    kde = KernelDensity(kernel=k, bandwidth=b)
    
    start = time.time()
    kde.fit(train_data)
    end = time.time()

    scores = kde.score_samples(val_data)
    threshold = np.quantile(scores, q)

    idxAnomaly = np.where(scores <= threshold)
    idxNormal = np.where(scores > threshold)

    val_predictions = scores

    val_predictions[idxAnomaly] = 1
    val_predictions[idxNormal] = 0

    conf_mat = confusion_matrix(val_labels, val_predictions)
    tn = conf_mat[0][0]
    tp = conf_mat[1][1]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    recall = recall_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)
    f1 = f1_score(val_labels, val_predictions)

    global df_kde
    df_kde = pd.concat([df_kde, pd.DataFrame( 
        {"kernel": [k],
        "bandwidth": [b],
        "quantile": [q],
        "accuracy": [accuracy],
        "recall" : [recall],
        "precision" : [precision],
        "f1" : [f1],
        "time" : [end - start]},
    dtype=object
    )], 
        ignore_index=True)

    df_kde.to_csv("kde.csv", index=False)
    
    return (k, b, q, accuracy, recall, precision, f1)

def process_kde():
    results_kde = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(train_kde)(k, b, q) for k in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'] for b in [0.5, 1.0, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0] for q in [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
    )       


# %%
from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem

df_nystroem_rbf = pd.DataFrame(
    {"gamma": [],
    "components_percent": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : [],
    "time" : []},
    dtype=object
)


def train_nystroem_rbf(g, n):
    
    feature_map_nystroem = Nystroem(kernel='rbf',
                                    gamma=g,
                                random_state=42,
                            n_components=int(n * len(train_data)))

    nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                            ("svm", linear_model.SGDOneClassSVM(random_state=42))])


    start = time.time()
    nystroem_approx_svm.fit(train_data)
    end = time.time()
    

    val_predictions = nystroem_approx_svm.predict(val_data)

    val_predictions[val_predictions == 1] = 0
    val_predictions[val_predictions == -1] = 1
    conf_mat = confusion_matrix(val_labels, val_predictions)
    tn = conf_mat[0][0]
    tp = conf_mat[1][1]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    recall = recall_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)
    f1 = f1_score(val_labels, val_predictions)
    
    global df_nystroem_rbf
    df_nystroem_rbf = pd.concat([df_nystroem_rbf, pd.DataFrame( 
        {"gamma": [g],
        "components_percent": [n],
        "accuracy": [accuracy],
        "recall" : [recall],
        "precision" : [precision],
        "f1" : [f1],
        "time" : [end - start]},
    dtype=object
    )], 
        ignore_index=True)
    
    df_nystroem_rbf.to_csv("nystroem-rbf.csv", index=False)
    
    return (g, n, accuracy, recall, precision, f1)

def process_nystroem_rbf():
    results_nystroem_rbf = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(train_nystroem_rbf)(g, n) for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0] for n in [10 ** -4, 10 ** -3, 10 ** -2, 4 * 10 ** -2, 5 * 10 ** -2, 10 ** -1, 3 * 10 ** -1]
    )       
    
# %%

df_nystroem_poly = pd.DataFrame(
    {"degree": [],
    "gamma": [],
    "components_percent": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : [],
    "time" : []},
    dtype=object
)

def train_nystroem_poly(d, g, n):
    
    feature_map_nystroem = Nystroem(kernel='poly',
                                            gamma=g,
                                            degree=d,
                                     random_state=42,
                                    n_components=int(n * len(train_data)))

    nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                            ("svm", linear_model.SGDOneClassSVM(random_state=42))])


    start = time.time()
    nystroem_approx_svm.fit(train_data)
    end = time.time()

    val_predictions = nystroem_approx_svm.predict(val_data)

    val_predictions[val_predictions == 1] = 0
    val_predictions[val_predictions == -1] = 1
    conf_mat = confusion_matrix(val_labels, val_predictions)
    tn = conf_mat[0][0]
    tp = conf_mat[1][1]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    recall = recall_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)
    f1 = f1_score(val_labels, val_predictions)

    global df_nystroem_poly
    df_nystroem_poly = pd.concat([df_nystroem_poly, pd.DataFrame( 
            {"degree": [d],
            "gamma": [g],
            "components_percent": [n],
            "accuracy": [accuracy],
            "recall" : [recall],
            "precision" : [precision],
            "f1" : [f1],
            "time" : [end - start]},
        dtype=object
        )], 
            ignore_index=True)

    df_nystroem_poly.to_csv("nystroem-poly.csv", index=False)

    
    return (d, g, n, accuracy, recall, precision, f1)

def process_nystroem_poly():
    results_nystroem_poly = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(train_nystroem_poly)(d, g, n) for d in range(1, 12, 1) for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0] for n in [10 ** -4, 10 ** -3, 10 ** -2, 4 * 10 ** -2, 5 * 10 ** -2, 10 ** -1, 3 * 10 ** -1]
    )       

# %%

df_nystroem_sigmoid = pd.DataFrame(
    {"gamma": [],
    "components_percent": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : [],
    "time" : []},
    dtype=object
)

def train_nystroem_sigmoid(g, n):
    
    feature_map_nystroem = Nystroem(kernel='sigmoid',
                                    gamma=g,
                                random_state=42,
                            n_components=int(n * len(train_data)))

    nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                            ("svm", linear_model.SGDOneClassSVM(random_state=42))])


    start = time.time()
    nystroem_approx_svm.fit(train_data)
    end = time.time()

    val_predictions = nystroem_approx_svm.predict(val_data)

    val_predictions[val_predictions == 1] = 0
    val_predictions[val_predictions == -1] = 1
    conf_mat = confusion_matrix(val_labels, val_predictions)
    tn = conf_mat[0][0]
    tp = conf_mat[1][1]
    fp = conf_mat[0][1]
    fn = conf_mat[1][0]
    accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
    recall = recall_score(val_labels, val_predictions)
    precision = precision_score(val_labels, val_predictions)
    f1 = f1_score(val_labels, val_predictions)
    
    global df_nystroem_sigmoid
    df_nystroem_sigmoid = pd.concat([df_nystroem_sigmoid, pd.DataFrame( 
        {"gamma": [g],
        "components_percent": [n],
        "accuracy": [accuracy],
        "recall" : [recall],
        "precision" : [precision],
        "f1" : [f1],
        "time" : [end - start]},
    dtype=object
    )], 
        ignore_index=True)
    
    df_nystroem_sigmoid.to_csv("nystroem-sigmoid.csv", index=False)
    
    return (g, n, accuracy, recall, precision, f1)

def process_nystroem_sigmoid():
    results_nystroem_sigmoid = Parallel(n_jobs=-1, require='sharedmem')(
        delayed(train_nystroem_sigmoid)(g, n) for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0] for n in [10 ** -4, 10 ** -3, 10 ** -2, 4 * 10 ** -2, 5 * 10 ** -2, 10 ** -1, 3 * 10 ** -1]
    )       
        

def process_model(idx):
    if idx == 0:
        process_ocsvm_rbf()
    elif idx == 1:
        process_ocsvm_poly()
    elif idx == 2:
        process_ocsvm_sigmoid()
    elif idx == 3:
        process_ocsvm_linear()
    elif idx == 4:
        process_gmm()
    elif idx == 5:
        process_kde()
    elif idx == 6:
        process_nystroem_rbf()
    elif idx == 7:
        process_nystroem_poly()
    elif idx == 8:
        process_nystroem_sigmoid()
        
if __name__ == '__main__': 
    # Parallel(n_jobs=-1, require='sharedmem')(
    #     delayed(process_model)(i) for i in range(9)
    # )
    
    for i in range(9):
        process_model(i)