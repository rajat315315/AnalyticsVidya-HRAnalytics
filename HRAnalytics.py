import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from scipy import stats

train = pd.read_csv(os.path.join('Data', 'train.csv'))
test = pd.read_csv(os.path.join('Data', 'test.csv'))

voting = True
metric = 'f1'
cv = 5

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
models_names = ['GradientBoostingClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier', 'ExtraTreeClassifier', 'KNeighborsClassifier', 'BernoulliNB']

params = [
    {'n_estimators':[10, 50, 100],'max_depth':[3, 10, 30]},
    {'n_estimators': [10, 50, 100], 'criterion':['entropy','gini'], 'max_depth': [None, 10, 20], 'n_jobs':[-1]},
    {'criterion':['entropy','gini'], 'max_depth': [None, 10, 20, 30]},
    {'criterion':['entropy','gini'], 'max_depth': [None, 10, 20, 30]},
    {'n_neighbors': [1, 3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'n_jobs':[-1]},
    {'alpha': [0.2, 0.5, 1.0], 'fit_prior':[True, False]},
]

best_models = []
print(concat)
with open("Output/Best_model_parameters.txt", "w") as f:
    for model, model_name, param in zip(models, models_names, params):


        gcv = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1, cv=cv, scoring=metric)
        gcv.fit( concat.iloc[:train.shape[0], :] , is_promoted)

        best_models.append(gcv.best_estimator_)
        print("For model : " + model_name)
        print("Best set of parameters : " + str(gcv.best_params_))
        print("Best " + metric + " : " + str(gcv.best_score_))
        f.write("For model : " + model_name + "\n")
        f.write("Best set of parameters : " + str(gcv.best_params_) + "\n")
        f.write("Best " + metric + " : "+ str(gcv.best_score_) + "\n")
        prediction = gcv.best_estimator_.predict(concat.iloc[train.shape[0]:, :])
        df = pd.DataFrame({"employee_id": employee_id, "is_promoted": prediction})
        df.to_csv("Best_" + model_name + "_output.csv", index=False)

if voting:
    vclf = VotingClassifier(estimators=list(zip(models_names, best_models)), voting='hard')
    vclf.fit(concat.iloc[:train.shape[0], :], is_promoted)
    prediction = vclf.predict(concat.iloc[train.shape[0]:, :])
    df = pd.DataFrame({"employee_id": employee_id, "is_promoted": prediction})
    df.to_csv("Output/voting_output.csv", index=False)



def calc_vote_from_files(files, index_col_name, value_col_name):

    index_col = pd.read_csv(files[0])[index_col_name]
    values = []
    for file in files:
        values.append(pd.read_csv(file)[value_col_name])

    final_values = stats.mode(np.array(values))[0][0]
    df = pd.DataFrame([index_col, final_values], columns=[index_col_name, value_col_name])
    df.to_csv("Voting_" + "_".join(files) + ".csv")