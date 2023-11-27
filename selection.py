import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LassoCV, ElasticNetCV

def select_k_best_features(x_data, y_data, k=0):
    
    best_features = SelectKBest(score_func = f_classif, k=k if k > 0 else 'all')
    calc_features = best_features.fit(x_data,y_data)
    #odabir najboljih feature-a
    usefull_features = pd.DataFrame({'attr_names' : x_data.columns.values, 'values' : calc_features.scores_})
    most_usefull_features = usefull_features.sort_values(by=['values'],ascending=False)
    if k > 0:
        most_usefull_features = most_usefull_features.head(k)

    return most_usefull_features

def train_best_classifier(
  classifier_name,
  classifier_model,
  parameters,
  x_train,
  y_train,
  x_test = None,
  y_test = None,
  verbose=False      
):
    grid_search_cv = GridSearchCV(
        classifier_model, 
        parameters, 
        verbose=1, 
        cv=10, 
        scoring='accuracy',
        n_jobs= 20
    )

    # optimize hyperparameters and determine the model with best results
    grid_search_cv.fit(x_train, y_train)
    best_model = grid_search_cv.best_estimator_ 
    print(grid_search_cv.best_estimator_)
    if verbose:
        y_preds = best_model.predict(x_test)
        print('Accuracy for {}: {:.2f}'.format(classifier_name, best_model.score(x_test, y_test)))
        print('Classification report for {}: '.format(classifier_name))
        print(classification_report(y_test, y_preds))

    return best_model

def calculate_classifier_metrics(
        classifier_models, 
        x_train,
        y_train):
    
    results_name = list()
    results_accuracy = list()
    results_precision = list()
    results_recall = list()
    results_f1 = list()
    names = list()
    for m in classifier_models:
        k_fold = StratifiedShuffleSplit(n_splits=10)
        cv_res = cross_validate(classifier_models[m], x_train, y_train, cv=k_fold, scoring=['accuracy', 'precision','recall','f1'])
        results_accuracy.append(cv_res['test_accuracy'].mean().round(3))
        results_precision.append(cv_res['test_precision'].mean().round(3))
        results_recall.append(cv_res['test_recall'].mean().round(3))
        results_f1.append(cv_res['test_f1'].mean().round(3))
        results_name.append(m)

    df_models = pd.DataFrame(
        {'Model name': results_name,
        'Accuracy': results_accuracy,
        'Precision': results_precision, 
        'Recall': results_recall,
        'F1': results_f1
        }
    )

    return df_models

def get_feature_scores_lasso(x_train, y_train):
    lasso = LassoCV(cv=10)                                                   
    lasso.fit(x_train, y_train)      
    lasso_values = lasso.coef_
    attr_names = x_train.columns.values
    lasso_features = pd.DataFrame({'attr_names':attr_names, 'values':lasso_values})
    lasso_features = lasso_features.sort_values(by='values', ascending=False)

    return lasso_features

def get_feature_scores_elenet(x_train, y_train):
    elenet = ElasticNetCV(cv=10)                                                   
    elenet.fit(x_train, y_train)      
    elenet_values = elenet.coef_
    attr_names = x_train.columns.values
    elenet_features = pd.DataFrame({'attr_names':attr_names, 'values':elenet_values})
    elenet_features = elenet_features.sort_values(by='values', ascending=False)

    return elenet_features


def bagging_feature_importance(classifier_model):
    if classifier_model.get_params()['max_features'] == 1:
        feature_importances = np.mean([tree.feature_importances_ for tree in classifier_model.estimators_], axis=0)
        return feature_importances
    else:
        raise Exception('Not support for max_feature <> 1')
    

def get_feature_importances(classifier_models, x_train, y_train):
    feature_importance = [classifier_models[m].feature_importances_ if m != 'bagging' else bagging_feature_importance(classifier_models[m]) for m in classifier_models]
    df_features = pd.DataFrame({'attr_names':x_train.columns, 'values':np.mean(feature_importance, axis=0)})
    return df_features.sort_values(by='values', ascending=False)




    

    
        
    