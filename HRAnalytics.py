import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

train = pd.read_csv(os.path.join('Data', 'train.csv'))
test = pd.read_csv(os.path.join('Data', 'test.csv'))

is_promoted = train.drop('is_promoted')
concat = pd.concat([train, test])

concat['previous_year_rating'] = concat.groupby('KPIs_met >80%')['previous_year_rating'].fillna('mean')
# concat.replace({'Bachelor\'s' : 1, 'Master\'s & above':2, 'Below Secondary': 0})


models = [GradientBoostingClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), ExtraTreeClassifier(), KNeighborsClassifier(), BernoulliNB()]


params = [
    {},
    {'n_estimators': [10, 50, 100], 'criterion':['entropy','gini'], 'max_depth': [None, 10, 20, 30], 'max_features': ['auto', 'sqrt', 'log2', None], 'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1], 'bootstrap':[True, False], 'n_jobs':[-1]},
    {'criterion':['entropy','gini'], 'max_depth': [None, 10, 20, 30], 'max_features': ['auto', 'sqrt', 'log2', None], 'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1]},
    {'criterion':['entropy','gini'], 'max_depth': [None, 10, 20, 30], 'max_features': ['auto', 'sqrt', 'log2', None], 'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1]},
    {'n_neighbors': [1, 3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'n_jobs':[-1]},
    {'alpha': [0.2, 0.5, 1.0], 'fit_prior':[True, False]},
]

