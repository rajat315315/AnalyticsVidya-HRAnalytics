from sklearn.preprocessing import MinMaxScaler

class MinMaxScaler_wrapper:

    scaler_objects = {}

    def fit(self, df, cols):

        for col in cols:
            df[col] = df[col] - df[col].mean()
            scaler = MinMaxScaler()
            self.scaler_objects[col] = scaler.fit(df[col].values.reshape(-1,1))


    def transform(self, df, cols):

        for col in cols:
            df[col] = self.scaler_objects[col].transform(df[col].values.reshape(-1,1))

    def fit_transform(self, df, cols):

        for col in cols:
            df[col] = df[col] - df[col].mean()
            scaler = MinMaxScaler()
            self.scaler_objects[col] = scaler.fit(df[col])
            df[col] = self.scaler_objects[col].transform(df[col])