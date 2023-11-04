from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class Utils:
    def __init__(self) -> None:
        pass 
    
    def performanceMetrics(self, val_predictions, val_labels):
        conf_mat = confusion_matrix(val_labels, val_predictions)
        tn = conf_mat[0][0]
        tp = conf_mat[1][1]
        fp = conf_mat[0][1]
        fn = conf_mat[1][0]
        accuracy = ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
        recall = recall_score(val_labels, val_predictions)
        precision = precision_score(val_labels, val_predictions)
        f1 = f1_score(val_labels, val_predictions)
        
        return accuracy, recall, precision, f1