import pandas as pd

class Blend():

    def Blend(self, level_one_classifiers, level_two_classifier, train_data, predictions, percent):
        self.level_one_classifiers =level_one_classifiers
        self.level_two_classifier =level_two_classifier
        self.train_data = train_data
        self.predictions = predictions
        self.percent = percent
        self.level_one_classifiers_fitted = []
        self.level_two_classifier_fitted

    def fit(self, train_data, predictions):

        level_one_train_data = train_data.iloc[ :int(self.percent / 100.0 * train_data.shape[0]), : ]
        level_one_predictions = predictions[ :int(self.percent / 100.0 * train_data.shape[0]) ]

        level_two_train_data = train_data.iloc[int(self.percent / 100.0 * train_data.shape[0]):, : ]
        level_two_predictions = predictions[ int(self.percent / 100.0 * train_data.shape[0]): ]

        level_one_output = pd.DataFrame()
        i=0

        for clf in self.level_one_classifiers:
            self.level_one_classifiers_fitted.append(clf.fit(level_one_train_data, level_one_predictions))

        for clf in self.level_one_classifiers_fitted:
            level_one_output[i] = clf.predict(level_two_train_data)
            i = i + 1

        self.level_two_classifier_fitted.fit(level_one_output, level_two_predictions)



    def predict(self, data):

        level_one_classifiers_predictions = pd.DataFrame()
        i=0

        for clf in self.level_one_classifiers_fitted:
            level_one_classifiers_predictions[i] = clf.predict(data)
            i = i + 1

        return self.level_two_classifier_fitted.predict(level_one_classifiers_predictions)