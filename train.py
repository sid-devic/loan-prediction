from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from read import X, y
from matplotlib import pyplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

X_scaled = preprocessing.scale(X)
print(X_scaled)
print('   ')
print(X_scaled.shape)

save_filename = 'model.sav'

def run_models(X_train, y_train, X_test, y_test, model_type = 'Non-balanced'):

    clfs = {'GradientBoosting': GradientBoostingClassifier(max_depth= 6, n_estimators=100, max_features = 0.3),
            'LogisticRegression' : LogisticRegression(),
            #'GaussianNB': GaussianNB(),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=10)
            }
    cols = ['model','matthews_corrcoef', 'roc_auc_score', 'precision_score', 'recall_score','f1_score']

    models_report = pd.DataFrame(columns = cols)
    conf_matrix = dict()

    for clf, clf_name in zip(clfs.values(), clfs.keys()):

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:,1]

        print('computing {} - {} '.format(clf_name, model_type))

        tmp = pd.Series({'model_type': model_type,
                         'model': clf_name,
                         'roc_auc_score' : metrics.roc_auc_score(y_test, y_score),
                         'matthews_corrcoef': metrics.matthews_corrcoef(y_test, y_pred),
                         'precision_score': metrics.precision_score(y_test, y_pred),
                         'recall_score': metrics.recall_score(y_test, y_pred),
                         'f1_score': metrics.f1_score(y_test, y_pred)})

        models_report = models_report.append(tmp, ignore_index = True)
        conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)

        plt.figure(1, figsize=(6,6))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('ROC curve - {}'.format(model_type))
        plt.plot(fpr, tpr, label = clf_name )
        plt.legend(loc=2, prop={'size':11})
    plt.plot([0,1],[0,1], color = 'black')

    return models_report, conf_matrix


'''
# without synthetically balanced classes
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.4, random_state=0)
models_report, conf_matrix = run_models(X_train, y_train, X_test, y_test, model_type = 'Non-balanced')

print(models_report)
'''

# with synthetic class balancing
index_split = int(len(X)/2)
X_train, y_train = SMOTE().fit_sample(X_scaled[0:index_split, :], y[0:index_split])
X_test, y_test = X_scaled[index_split:], y[index_split:]
logi = LogisticRegression()
logi.fit(X_train, y_train)

y_pred = logi.predict(X_test)
y_score = logi.predict_proba(X_test)[:,1]
model_type = "Balanced"
print('computing {} - {} '.format('Logistic Regression', model_type))

tmp = pd.Series({'model_type': model_type,
                'model': 'Logistic Regression',
                'roc_auc_score' : metrics.roc_auc_score(y_test, y_score),
                'matthews_corrcoef': metrics.matthews_corrcoef(y_test, y_pred),
                'precision_score': metrics.precision_score(y_test, y_pred),
                'recall_score': metrics.recall_score(y_test, y_pred),
                'f1_score': metrics.f1_score(y_test, y_pred)})

#models_report = models_report.append(tmp, ignore_index = True)
#conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)

plt.figure(1, figsize=(6,6))
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC curve - {}'.format(model_type))
plt.plot(fpr, tpr, label = 'Logistic Regression' )
plt.legend(loc=2, prop={'size':11})
plt.show
#scores = cross_val_score(clf, X_scaled, y , cv=5, scoring='roc_auc')

# models_report_bal, conf_matrix_bal = run_models(X_train, y_train, X_test, y_test, model_type = 'Balanced')
# print(models_report)

pickle.dump(logi, open(save_filename, 'wb'))

