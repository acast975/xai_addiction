import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def show_value_distribution_per_column(data, column_name):
    labels = pd.unique(data[column_name])
    countByClass = data.groupby(column_name).size()
    plt.pie(countByClass, labels = labels, autopct='%.0f%%')
    plt.show()

def show_selected_features_old(features):
    sns.set()
    plt.title('Feature selection using Pearson\'s coefficient')
    plt.xlabel('Feature score')
    plt.ylabel('Feature name')

    sns.barplot(y=features['feature'], x=features['score'])
    plt.show()

def show_selected_features(title, label_x, label_y, data_x, data_y):
    sns.set()
    plt.figure(figsize=(10,17))
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    sns.barplot(y=data_y, x=data_x)
    plt.show()

def add_labels(x,y):
    for i in range(len(x)):
        plt.text(x[i], y[i] , y[i], ha = 'center')

def show_model_comparasion(df_models):
    no_models = len(df_models['Model name'])
    barWidth = 0.20
    fig = plt.subplots(figsize =(15, 7))
    metrics = ['Accuracy','Precision','Recall','F1']
    plt.xlabel('Model', fontweight ='bold', fontsize = 15)
    plt.ylabel('Values', fontweight ='bold', fontsize = 15)

    br1 = np.arange(no_models)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x +barWidth for x in br3]

    plt.xticks([r + barWidth for r in range(no_models)],
            df_models['Model name'])

    add_labels(br1,df_models['Accuracy'])
    plt.bar(br1, df_models['Accuracy'], color ='r', width = barWidth,
            edgecolor ='grey', label ='Accuracy')

    add_labels(br2, df_models['Precision'])
    plt.bar(br2, df_models['Precision'], color ='g', width = barWidth,
            edgecolor ='grey', label ='Precision')

    add_labels(br3, df_models['Recall'])
    plt.bar(br3, df_models['Recall'], color ='b', width = barWidth,
            edgecolor ='grey', label ='Recall')

    add_labels(br4, df_models['F1'])
    plt.bar(br4, df_models['F1'], color ='y', width = barWidth,
            edgecolor ='grey', label ='F1')

    plt.legend()
    plt.show()

def highlight_cells(val):
    color = 'yellow' if val == True else ''
    return 'background-color: {}'.format(color)

def show_genetic_selection_results(classifier_model, x_train):
    gen_opt_features = classifier_model.support_
    gen_df = pd.DataFrame({'features': x_train.columns, 'is_used':gen_opt_features})

    selected_features = gen_df[gen_df['is_used'] == True]['features']
    gen_df.style.applymap(highlight_cells)
    print(gen_df)




