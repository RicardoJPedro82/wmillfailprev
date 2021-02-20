from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from termcolor import colored

class CustomStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mean_ = {}
        self.std_ = {}

    def fit(self, X, y=None):

        for turbine in ['T01','T06','T07','T09','T11']:
            X_filter = X[X['Turbine_ID']==turbine]
            X_filter = X_filter.drop(columns='Turbine_ID').to_numpy()
            self.mean_[turbine] = X_filter.mean(axis=0)
            self.std_[turbine] = X_filter.std(axis=0)

        return self #.mean_,self.std_

    def transform(self, X, y=None):
        X_turbine={}

        for turbine in ['T01','T06','T07','T09','T11']:

            X_turbine[turbine] = (X[X['Turbine_ID']==turbine].drop(columns='Turbine_ID').to_numpy() - self.mean_[turbine]) / self.std_[turbine]

        X_turbine_all=X_turbine['T01']

        for turbine in ['T06','T07','T09','T11']:

            X_turbine_all=np.concatenate([X_turbine_all,X_turbine[turbine]])

        return X_turbine_all

class Trainer():
    def __init__(self, x_train, y_train, **kwargs):

        self.pipeline = None
        self.kwargs = kwargs
        self.x_train = x_train
        self.y_train = y_train
        self.component = self.kwargs.get("component")

    def get_estimator(self):
        """return estimator"""
        if self.component == "df_generator":
            model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                        metric_params=None, n_jobs=-1, n_neighbors=5,
                                        p=2, weights='uniform')
        elif self.component == "df_hydraulic":
            model = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                                        intercept_scaling=1, max_iter=1000, multi_class='warn',
                                        n_jobs=-1, penalty='l2', solver='lbfgs',
                                        tol=0.0001, verbose=0, warm_start=False)
        elif self.component == "df_transformer":
            model = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
                                        learning_rate=1.0, n_estimators=200)
        elif self.component == "df_gen_bear":
            model = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                        metric_params=None, n_jobs=-1, n_neighbors=5, p=2,
                                        weights='distance')
        elif self.component == "df_gearbox":
            model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                        max_depth=None, max_features='auto', max_leaf_nodes=None,
                                        min_impurity_decrease=0.0, min_impurity_split=None,
                                        min_samples_leaf=1, min_samples_split=2,
                                        min_weight_fraction_leaf=0.0, n_estimators=750, n_jobs=-1,
                                        oob_score=False, verbose=0, warm_start=False)
        else:
            print('Invalid Component')

        return model

    def train(self):
        """train the selected model"""
        model = self.get_estimator()
        model.fit(self.x_train, self.y_train)
