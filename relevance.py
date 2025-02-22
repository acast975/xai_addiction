import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier
from eli5 import explain_prediction_dfs

def get_predictions(classifier_name, x_train, y_train, x_test, y_test):
    
    if classifier_name == 'random_forest':
        model = RandomForestClassifier(criterion='entropy', max_depth=9, n_estimators=40)
    elif classifier_name == 'xgboost':
        model = XGBClassifier(max_depth=9, learning_rate=0.1, n_estimators=50)
    else:
        model = DecisionTreeClassifier(criterion='entropy', max_depth=5, max_leaf_nodes=7)
    model.fit(x_train, y_train)
    print('Train accuracy: {0:.2f}'.format(model.score(x_train, y_train)))
    print('Test accuracy: {0:.2f}'.format(model.score(x_test, y_test)))
    y_preds = model.predict(x_test)
    print('Confusion Matrix: ')
    print(confusion_matrix(y_test, y_preds))
    print()
    print('Classification Report:')
    print(classification_report(y_test, y_preds))

    misclassified=y_test!=y_preds
    pred_tbl=pd.DataFrame({'Expected':y_test,'Predicted':y_preds,'Misclassified':misclassified})
    
    predictions = dict(
        model = model,
        predicted_values = pred_tbl,
        tp = np.where((pred_tbl.Misclassified==False) & (pred_tbl.Expected==1)),
        tn = np.where((pred_tbl.Misclassified==False) & (pred_tbl.Expected==0)),
        fp = np.where((pred_tbl.Misclassified==True) & (pred_tbl.Expected==0)),
        fn = np.where((pred_tbl.Misclassified==True) & (pred_tbl.Expected==1))
    )

    return predictions

def get_shap_values_as_data_frame(data, shap_values, class_label = -1):
    # https://stackoverflow.com/questions/65534163/get-a-feature-importance-from-shap-values
    columns = data.columns.tolist()
    
    c_idxs = []
    for column in columns: c_idxs.append(data.columns.get_loc(column))  # Get column locations for desired columns in given dataframe
    if isinstance(shap_values, list):   # If shap values is a list of arrays (i.e., several classes)
        if class_label == -1:
            means = [np.abs(shap_values[class_][:, c_idxs]).mean(axis=0) for class_ in range(len(shap_values))]  # Compute mean shap values per class 
        else:
            means = [np.abs(shap_values[class_label][:, c_idxs]).mean(axis=0)]
        shap_means = np.sum(np.column_stack(means), 1)  # Sum of shap values over all classes 
    else:                               # Else there is only one 2D array of shap values
        assert len(shap_values.shape) == 2, 'Expected two-dimensional shap values array.'
        shap_means = np.abs(shap_values).mean(axis=0)
    
    # Put into dataframe along with columns and sort by shap_means, reset index to get ranking
    df_ranking = pd.DataFrame({'attr_names': columns, 'values': shap_means}).sort_values(by='values', ascending=False).reset_index(drop=True)
    df_ranking.index += 1
    return df_ranking

def add_top_features_count(name, df_top_features_count, df_top, n=0):
    top = df_top
    if name != 'genetic_selection':
        top['values'] = df_top['values'].abs()
        top = top.sort_values(by=['values'],ascending=False)
        if n > 0:
            top = top.head(n)
    else:
        top = df_top
        top = top.sort_values(by=['values'],ascending=False)
        top = top[top.values==False]
    
    df_ret = df_top_features_count
    df_ret.top_count = df_top_features_count.top_count.add(df_top_features_count.attr_names.isin(top.attr_names).astype(int))
    if name == 'SHAP' or name == 'eli5' or name=='tree_importance':
        df_ret.top_count_xai = df_top_features_count.top_count_xai.add(df_top_features_count.attr_names.isin(top.attr_names).astype(int))
    else:
        df_ret.top_count_selection = df_top_features_count.top_count_selection.add(df_top_features_count.attr_names.isin(top.attr_names).astype(int))

    return df_ret

def add_top_features_interpretation_score(df_top, df_interpretation):
    df_temp = df_interpretation[df_interpretation.values>0]
    df_top.score = df_top.score.add(df_top.attr_names.isin(df_temp).astype(int))
    return df_top

def get_important_features_statistics(predictions, x_test, top_n_features):
    
    for pred_type in ['tp', 'tn', 'fp', 'fn']:
        df_top_n_features = pd.DataFrame({'attr_names' : top_n_features})
        df_top_n_features['score'] = 0
        n_features = list()

        for i in predictions[pred_type][0]:
            df = explain_prediction_dfs(
                predictions['model'],
                x_test.iloc[i,:],
                targets=[0, 1], 
                target_names=['PIU no', 'PIU yes'],
                feature_names=x_test.columns.values.tolist(),
                top=len(x_test.columns)+1
            )
            df_interpretation = pd.DataFrame({'attr_names' : df['targets'].iloc[:, 1], 'values' : df['targets'].iloc[:, 2]})
            df_interpretation = df_interpretation[df_interpretation['values']>0].iloc[:11]
            n_features.append(len(df_interpretation[df_interpretation.attr_names.isin(df_top_n_features.attr_names)]))
            df_top_n_features.score = df_top_n_features.score.add(df_top_n_features.attr_names.isin(df_interpretation.attr_names).astype(int))

        df_top_n_features = df_top_n_features.sort_values(by=['score'], ascending=False)
        print(df_top_n_features)
        print('Number of {0} instances {1:.2f}. Mean number of relevant features per classification result {2:.2f}.'.format(pred_type, len(n_features), np.mean(np.array(n_features))))

    
