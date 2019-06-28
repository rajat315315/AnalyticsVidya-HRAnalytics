import pandas as pd

def Pre_Process(df, config):


    #Missing value Imputation

    # Plain
    for col, ImputationType in config['IMPUTATION']['PLAIN']:

        if ImputationType == "MEAN":
            df[col] = df[col].fillna(df[col].mean())
        elif ImputationType == "MODE":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(ImputationType)


    # Grouped Imputation
    for col, Groups, method in config['IMPUTATION']['GROUPED']:

        if method == "MEAN":
            df[col] = df.groupby(Groups)[col].apply(lambda x: x.fillna(x.mean()))
        elif method == "MODE":
            df[col] = df.groupby(Groups)[col].apply(lambda x: x.fillna(x.mode()[0]))
        else:
            df[col] = df.groupby(Groups)[col].apply(lambda x: x.fillna(ImputationType))

    # Replace Data
    for col, replacement_dict in config['REPLACE']:

        df[col] = df[col].replace(replacement_dict)
