import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

train = pd.read_csv(os.path.join('Data', 'train.csv'))
test = pd.read_csv(os.path.join('Data', 'test.csv'))

is_promoted = train['is_promoted']
employee_id = test['employee_id']
train = train.drop(['is_promoted', ], 1)
concat = pd.concat([train, test], ignore_index=True)

# concat.replace({'Bachelor\'s' : 1.0, 'Master\'s & above':2.0, 'Below Secondary': 0.0}, inplace=True)
concat['previous_year_rating'] = concat.groupby('KPIs_met >80%')['previous_year_rating'].apply(lambda x: x.fillna(x.mean()))
# concat = concat.dropna(subset=['education'], axis=0)

concat = pd.concat([concat, pd.get_dummies(concat['department'])], 1)
concat = pd.concat([concat, pd.get_dummies(concat['region'])], 1)
concat = pd.concat([concat, pd.get_dummies(concat['gender'])], 1)
concat = pd.concat([concat, pd.get_dummies(concat['recruitment_channel'])], 1)
concat = concat.drop(['education', 'department', 'recruitment_channel', 'gender', 'region', 'employee_id'], 1)

models = [GradientBoostingClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), ExtraTreeClassifier(), KNeighborsClassifier(), BernoulliNB()]


params = [
    {'n_estimators':[10, 50, 100],'max_depth':[3, 10, 30]},
    {'n_estimators': [10, 50, 100], 'criterion':['entropy','gini'], 'max_depth': [None, 10, 20], 'n_jobs':[-1]},
    {'criterion':['entropy','gini'], 'max_depth': [None, 10, 20, 30]},
    {'criterion':['entropy','gini'], 'max_depth': [None, 10, 20, 30]},
    {'n_neighbors': [1, 3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'n_jobs':[-1]},
    {'alpha': [0.2, 0.5, 1.0], 'fit_prior':[True, False]},
]

print(concat)
# for model, param in zip(models, params):
#
#
#     gcv = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1, cv=5, scoring='f1')
#     gcv.fit( concat.iloc[:train.shape[0], :] , is_promoted)
#
#     gcv.best_estimator_
    # print(gcv.best_params_)
    # print(gcv.best_score_)

clf = RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=20, max_features=None, n_estimators=50, n_jobs=-1)
clf.fit(concat.iloc[:train.shape[0], :] , is_promoted)
prediction = clf.predict(concat.iloc[train.shape[0]:, :])

df = pd.DataFrame({"employee_id": employee_id, "is_promoted": prediction})
df.to_csv("output.csv", index=False)
