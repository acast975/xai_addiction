import numpy as np
import pandas as pd

def get_predictions(classifier_model, x_test, y_test):
    y_preds = classifier_model.predict(x_test)
    misclassified=y_test!=y_preds
    pred_tbl=pd.DataFrame({'Expected':y_test,'Predicted':y_preds,'Misclassified':misclassified})
    
    predictions = dict(
        predicted_values = pred_tbl,
        tp = np.where((pred_tbl.Misclassified==False) & (pred_tbl.Expected==1)),
        tn = np.where((pred_tbl.Misclassified==False) & (pred_tbl.Expected==0)),
        fp = np.where((pred_tbl.Misclassified==True) & (pred_tbl.Expected==0)),
        fn = np.where((pred_tbl.Misclassified==True) & (pred_tbl.Expected==1))
    )

    return predictions