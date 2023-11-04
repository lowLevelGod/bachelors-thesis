from pathlib import Path 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

class DataLoader:
    def __init__(self) -> None:
        data = pd.read_csv(Path("./creditcard.csv"))
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
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        self.test_labels = test_labels
        self.val_labels = val_labels
    
    def loadData(self):
        return self.train_data, self.val_data, self.test_data
    def loadLabels(self):
        return self.val_labels, self.test_labels
        