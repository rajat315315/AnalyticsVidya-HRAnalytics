import pandas as pd
import numpy as np
import os
import time
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense

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
#
# models = [GradientBoostingClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), ExtraTreeClassifier(), KNeighborsClassifier(), BernoulliNB()]
# models_names = ['GradientBoostingClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier', 'ExtraTreeClassifier', 'KNeighborsClassifier', 'BernoulliNB']
#
# params = [
#     {'n_estimators':[10, 50, 100],'max_depth':[3, 10, 30]},
#     {'n_estimators': [10, 50, 100], 'criterion':['entropy','gini'], 'max_depth': [None, 10, 20], 'n_jobs':[-1]},
#     {'criterion':['entropy','gini'], 'max_depth': [None, 10, 20, 30]},
#     {'criterion':['entropy','gini'], 'max_depth': [None, 10, 20, 30]},
#     {'n_neighbors': [1, 3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'n_jobs':[-1]},
#     {'alpha': [0.2, 0.5, 1.0], 'fit_prior':[True, False]},
# ]
#
# best_models = []
# print(concat)
# with open("Output/Best_model_parameters.txt", "w") as f:
#     for model, model_name, param in zip(models, models_names, params):
#
#
#         gcv = GridSearchCV(estimator=model, param_grid=param, n_jobs=-1, cv=cv, scoring=metric)
#         gcv.fit( concat.iloc[:train.shape[0], :] , is_promoted)
#
#         best_models.append(gcv.best_estimator_)
#         print("For model : " + model_name)
#         print("Best set of parameters : " + str(gcv.best_params_))
#         print("Best " + metric + " : " + str(gcv.best_score_))
#         f.write("For model : " + model_name + "\n")
#         f.write("Best set of parameters : " + str(gcv.best_params_) + "\n")
#         f.write("Best " + metric + " : "+ str(gcv.best_score_) + "\n")
#         prediction = gcv.best_estimator_.predict(concat.iloc[train.shape[0]:, :])
#         df = pd.DataFrame({"employee_id": employee_id, "is_promoted": prediction})
#         df.to_csv("Best_" + model_name + "_output.csv", index=False)
#
# if voting:
#     vclf = VotingClassifier(estimators=list(zip(models_names, best_models)), voting='hard')
#     vclf.fit(concat.iloc[:train.shape[0], :], is_promoted)
#     prediction = vclf.predict(concat.iloc[train.shape[0]:, :])
#     df = pd.DataFrame({"employee_id": employee_id, "is_promoted": prediction})
#     df.to_csv("Output/voting_output.csv", index=False)

def make_model(hidden, n_cols, output):

    model = Sequential()
    model.add(Dense(hidden, activation = 'relu', input_shape=(n_cols, )))
    model.add(Dense(output, activation = 'sigmoid'))

    return model


def vary_threshold(prediction_file, index_col_name, value_col_name, init, upto):

    df1 = pd.read_csv(prediction_file)
    for thresh in np.arange(init, upto, 0.05):
        df = df1.__deepcopy__()
        df.loc[df[value_col_name] >= thresh, value_col_name] = 1
        df.loc[df[value_col_name] < thresh, value_col_name] = 0
        df[value_col_name] = df[value_col_name].astype(int)
        df.to_csv("Output/DNN_" + str(int(np.sqrt(concat.shape[1]))) + "_" + str(time.time()) + "_threshold_" + str(thresh) + "_.csv", index=False)


def calc_vote_from_files(files, index_col_name, value_col_name):

    index_col = pd.read_csv("Output/" + files[0])[index_col_name]
    values = []
    for file in files:
        values.append(pd.read_csv("Output/" + file)[value_col_name])

    final_values = stats.mode(np.array(values))[0][0]
    df = pd.DataFrame({index_col_name : index_col, value_col_name: final_values})
    df.to_csv("Output/Voting_" + "_".join(files) + ".csv", index=False)


files = ['Best_GradientBoostingClassifier_output.csv', 'Best_RandomForestClassifier_output.csv', 'DNN_7_1559382607.6679935_threshold_0.45_.csv']
index_col_name = 'employee_id'
value_col_name = 'is_promoted'
# calc_vote_from_files(files, index_col_name, value_col_name)



# Running DNN

model = make_model(int(np.sqrt(concat.shape[1])), concat.shape[1], 1)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(concat.iloc[:train.shape[0], :] , is_promoted, epochs=50)
prediction = pd.DataFrame(model.predict(concat.iloc[train.shape[0]: , :] ))
index_col = pd.read_csv("Output/" + files[0])[index_col_name]
df = pd.DataFrame({index_col_name : index_col, value_col_name: prediction[0].values})
df.to_csv("Output/DNN_" + str(int(np.sqrt(concat.shape[1]))) + "_.csv", index=False)

vary_threshold("Output/DNN_" + str(int(np.sqrt(concat.shape[1]))) + "_.csv", index_col_name, value_col_name, 0.05, 0.60)