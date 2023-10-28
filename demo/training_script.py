# %%
from pathlib import Path
import numpy as np 
import pandas as pd 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn import pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc

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
df = pd.DataFrame(
    {"gamma": [],
    "nu": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0]:
    for n in [10 ** -4, 10 ** -3, 5 * 10 ** -3, 7 * 10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9 ]:
        ocsvm = OneClassSVM(kernel='rbf', gamma=g, nu=n)
        ocsvm.fit(train_data)
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
        
        df = pd.concat([df, pd.DataFrame( 
            {"gamma": [g],
            "nu": [n],
            "accuracy": [accuracy],
            "recall" : [recall],
            "precision" : [precision],
            "f1" : [f1]},
        dtype=object
        )], 
            ignore_index=True)
        
        df.to_csv("ocsvm-rbf.csv", index=False)
        

# %%
df = pd.DataFrame(
    {"degree": [],
    "gamma": [],
    "nu": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)


for d in range(1, 12, 1):
    for g in [0.03, 0.5, 1.0, 2.0]:
        for n in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:            
            ocsvm = OneClassSVM(kernel='poly', degree=d, gamma=g, nu=n)
            ocsvm.fit(train_data)
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

            df = pd.concat([df, pd.DataFrame( 
                {"degree": [d],
                "gamma": [g],
                "nu": [n],
                "accuracy": [accuracy],
                "recall" : [recall],
                "precision" : [precision],
                "f1" : [f1]},
            dtype=object
            )], 
                ignore_index=True)

            df.to_csv("ocsvm-poly.csv", index=False)


# %%
df = pd.DataFrame(
    {"gamma": [],
    "nu": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for g in [0.03, 0.5, 1.0, 2.0]:
    for n in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:    
        ocsvm = OneClassSVM(kernel='sigmoid',gamma=g, nu=n)
        ocsvm.fit(train_data)
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

        df = pd.concat([df, pd.DataFrame( 
            {"gamma": [g],
            "nu": [n],
            "accuracy": [accuracy],
            "recall" : [recall],
            "precision" : [precision],
            "f1" : [f1]},
        dtype=object
        )], 
            ignore_index=True)

        df.to_csv("ocsvm-sigmoid.csv", index=False)


# %%
df = pd.DataFrame(
    {
    "nu": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for n in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]:    
    ocsvm = OneClassSVM(kernel='linear',  nu=n)
    ocsvm.fit(train_data)
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

    df = pd.concat([df, pd.DataFrame( 
        {
        "nu": [n],
        "accuracy": [accuracy],
        "recall" : [recall],
        "precision" : [precision],
        "f1" : [f1]},
    dtype=object
    )], 
        ignore_index=True)

    df.to_csv("ocsvm-linear.csv", index=False)


# %%
from sklearn.mixture import GaussianMixture

df = pd.DataFrame(
    {"components": [],
    "quantile": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for n in [i for i in range(1, 21)]:
    for q in [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]:
        gmm = GaussianMixture(n_components=n, max_iter=10 ** 5, tol=10 ** -5, random_state=42)
        gmm.fit(train_data)
        
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
        
        df = pd.concat([df, pd.DataFrame( 
            {"components": [n],
            "quantile": [q],
            "accuracy": [accuracy],
            "recall" : [recall],
            "precision" : [precision],
            "f1" : [f1]},
        dtype=object
        )], 
            ignore_index=True)
        
        df.to_csv("gmm.csv", index=False)



# %%
from sklearn.neighbors import KernelDensity

df = pd.DataFrame(
    {"kernel": [],
    "bandwidth": [],
    "quantile": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for k in ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']:
    for b in [0.5, 1.0, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0]:
        for q in [10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]:
            kde = KernelDensity(kernel=k, bandwidth=b)
            kde.fit(train_data)

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

            df = pd.concat([df, pd.DataFrame( 
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

            df.to_csv("kde.csv", index=False)


# %%
from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem


df = pd.DataFrame(
    {"gamma": [],
    "components_percent": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0]:
    for n in [10 ** -4, 10 ** -3, 10 ** -2, 4 * 10 ** -2, 5 * 10 ** -2, 10 ** -1, 3 * 10 ** -1]:
        feature_map_nystroem = Nystroem(kernel='rbf',
                                        gamma=g,
                                 random_state=42,
                                n_components=int(n * len(train_data)))

        nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                                ("svm", linear_model.SGDOneClassSVM(random_state=42))])


        nystroem_approx_svm.fit(train_data)

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
        
        df = pd.concat([df, pd.DataFrame( 
            {"gamma": [g],
            "components_percent": [n],
            "accuracy": [accuracy],
            "recall" : [recall],
            "precision" : [precision],
            "f1" : [f1]},
        dtype=object
        )], 
            ignore_index=True)
        print(n)
        df.to_csv("nystroem-rbf.csv", index=False)
        

# %%

df = pd.DataFrame(
    {"degree": [],
    "gamma": [],
    "components_percent": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for d in range(1, 12, 1):
    for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0]:
        for n in [10 ** -4, 10 ** -3, 10 ** -2, 4 * 10 ** -2, 5 * 10 ** -2, 10 ** -1, 3 * 10 ** -1]:
            feature_map_nystroem = Nystroem(kernel='poly',
                                            gamma=g,
                                            degree=d,
                                     random_state=42,
                                    n_components=int(n * len(train_data)))

            nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                                    ("svm", linear_model.SGDOneClassSVM(random_state=42))])


            nystroem_approx_svm.fit(train_data)

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

            df = pd.concat([df, pd.DataFrame( 
                {"degree": [d],
                "gamma": [g],
                "components_percent": [n],
                "accuracy": [accuracy],
                "recall" : [recall],
                "precision" : [precision],
                "f1" : [f1]},
            dtype=object
            )], 
                ignore_index=True)

            df.to_csv("nystroem-poly.csv", index=False)


# %%


df = pd.DataFrame(
    {"gamma": [],
    "components_percent": [],
    "accuracy": [],
    "recall" : [],
    "precision" : [],
    "f1" : []},
    dtype=object
)

for g in [10 ** -3, 10 ** -2, 2 * 10 ** -2, 0.03, 0.5, 1.0, 2.0]:
    for n in [10 ** -4, 10 ** -3, 10 ** -2, 4 * 10 ** -2, 5 * 10 ** -2, 10 ** -1, 3 * 10 ** -1]:
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
        conf_mat = confusion_matrix(val_labels, val_predictions)
        tn = conf_mat[0][0]
        tp = conf_mat[1][1]
        fp = conf_mat[0][1]
        fn = conf_mat[1][0]
        accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
        recall = recall_score(val_labels, val_predictions)
        precision = precision_score(val_labels, val_predictions)
        f1 = f1_score(val_labels, val_predictions)
        
        df = pd.concat([df, pd.DataFrame( 
            {"gamma": [g],
            "components_percent": [n],
            "accuracy": [accuracy],
            "recall" : [recall],
            "precision" : [precision],
            "f1" : [f1]},
        dtype=object
        )], 
            ignore_index=True)
        
        df.to_csv("nystroem-sigmoid.csv", index=False)
        


