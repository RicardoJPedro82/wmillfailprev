
#Guia para utilização da função final get_X_y:

#1º argumento: dataframe
#2º argumento: número de sequencias a gerar para cada uma das turbinas
#3º argumento: comprimento de cada uma das sequências (nº de timestamps recolhidos para formar uma sequencia)
#4º argumento: especificar a variavel target

import pandas as pd
import numpy as np

def subsample_sequence(df, length, turbine):

    last_possible = df[df['Turbine_ID']==turbine].shape[0] - length

    random_start = np.random.randint(0, last_possible)
    df_sample = df[df['Turbine_ID']==turbine][random_start: random_start+length]

    return df_sample



def split_subsample_sequence(df, length, target, turbine):


    df_subsample = subsample_sequence(df, length, turbine)
    y_sample = df_subsample.iloc[df_subsample.shape[0]-1][target]


    X_sample = df_subsample[0:df_subsample.shape[0]].drop(columns=[target,'Turbine_ID','Timestamp'])
    X_sample = X_sample.values

    return np.array(X_sample), np.array(y_sample)



def get_X_y(df, number_of_sequences, length, target):
    X, y = [], []

    for turbine in ['T01','T06', 'T07', 'T09', 'T11']:

        for i in range(number_of_sequences):
            xi, yi = split_subsample_sequence(df, length, target, turbine)
            X.append(xi)
            y.append(yi)

    X = np.array(X)
    y = np.array(y)

    return X, y

