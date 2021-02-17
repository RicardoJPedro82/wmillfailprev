from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mean_ = {}
        self.std_ = {}

    def fit(self, X):

        for turbine in ['T01','T06','T07','T09','T11']:
            X_filter = X[X['Turbine_ID']==turbine]
            X_filter = X_filter.drop(columns='Turbine_ID').to_numpy()
            self.mean_[turbine] = X_filter.mean(axis=0)
            self.std_[turbine] = X_filter.std(axis=0)



        return self.mean_,self.std_

    def transform(self, X):
        X_turbine={}



        for turbine in ['T01','T06','T07','T09','T11']:

            X_turbine[turbine] = (X[X['Turbine_ID']==turbine].drop(columns='Turbine_ID').to_numpy() - self.mean_[turbine]) / self.std_[turbine]

        X_turbine_all=X_turbine['T01']

        for turbine in ['T06','T07','T09','T11']:

            X_turbine_all=np.concatenate([X_turbine_all,X_turbine[turbine]])

        return X_turbine_all
