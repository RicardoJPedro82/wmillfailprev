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
        # self.model = model

    def get_estimator(self):
        """return estimator"""
        if self.component == "df_generator":
            model =  KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=None, n_neighbors=5, p=2,weights='uniform')
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
        self.model = self.get_estimator()
        self.model.fit(self.x_train, self.y_train)
        return self.model

    def predict(self, x_test):
        """Get the predictions of the selected model"""
        return self.model.predict(x_test)



# Aplicação de métricas de negócio ás previsões para poder determinar poupanças

def met_poupanca_TP(real, prev):
    if real == prev and real ==1:
        return 1
    else:
        return 0

def met_poupanca_TN(real, prev):
    if real == prev and real == 0:
        return 1
    else:
        return 0

def met_poupanca_FP(real, prev):
    if real == 0 and prev == 1:
        return 1
    else:
        return 0

def met_poupanca_FN(real, prev):
    if real == 1 and prev == 0:
        return 1
    else:
        return 0

def metrics_create_df(df_test_in, y_test_in, y_pred_in, component, days=10):
    'Criar o dataframe de avaliação dos resultados da predição'
    "df_gearbox, df_generator, df_gen_bear, df_transformer, df_hydraulic"
    cols= ['Date','Turbine_ID', 'TTF']
    met_cre_df = df_test_in[cols].copy()
    met_cre_df['TTF'] = np.ceil(met_cre_df['TTF'])
    met_cre_df['y_test'] = y_test_in
    met_cre_df['y_pred'] = y_pred_in
    met_cre_df = met_cre_df.reset_index().drop(columns='index')
    met_cre_df['TP'] = [met_poupanca_TP(met_cre_df.y_test[i], met_cre_df.y_pred[i]) for i in range(len(met_cre_df.y_pred))]
    met_cre_df['TN'] = [met_poupanca_TN(met_cre_df.y_test[i], met_cre_df.y_pred[i]) for i in range(len(met_cre_df.y_pred))]
    met_cre_df['FP'] = [met_poupanca_FP(met_cre_df.y_test[i], met_cre_df.y_pred[i]) for i in range(len(met_cre_df.y_pred))]
    met_cre_df['FN'] = [met_poupanca_FN(met_cre_df.y_test[i], met_cre_df.y_pred[i]) for i in range(len(met_cre_df.y_pred))]

    #rolling de 10 dias
    met_cre_df['new_FP']= met_cre_df['FP'].rolling(days, min_periods=1).sum()

    #'Dataframe com as falhas'
    falhas = met_cre_df[met_cre_df.TTF==1]

    #'Lista de indices com as falhas'
    falhas_index_list = []
    dias_primeiro_TP = []
    for i in range(len(falhas)):
        falhas_index_list.append(falhas.index[i])
    #'Criação do dicionário para amazenar as metricas da confusão'
    cf_numbers = {'TP': 0,'TN': 0,'FP': 0,'FN': 0,}

    # Para as ocorrências
    ocorrencias = met_cre_df[met_cre_df['y_test'] == 1]
    for ind in range(len(falhas_index_list)):
        for key in cf_numbers:
            if ind == 0:
                cf_numbers[key] += ocorrencias.loc[:falhas_index_list[ind]][key].sum()
            else:
                cf_numbers[key] += ocorrencias.loc[falhas_index_list[ind-1]:falhas_index_list[ind]][key].sum()
        if cf_numbers['TP'] >= 1:
            cf_numbers['TP'] = 1
            cf_numbers['FN'] = 0
        elif cf_numbers['FN'] >= 1:
            cf_numbers['TP'] = 0
            cf_numbers['FN'] = 1
        if len(ocorrencias[ocorrencias.TP == 1]) != 0:
            dias_primeiro_TP.append(ocorrencias.loc[ocorrencias[ocorrencias.TP == 1].index[ind]].TTF.astype(int))

    # Para as não ocorrências
    ocorrencias = met_cre_df[met_cre_df['y_test'] == 0]

    cf_numbers['TN'] = ocorrencias.TN.sum()

    #Calcular false positives de mercado
    new_fp_df = met_cre_df[met_cre_df['FP']==1]
    cf_numbers['FP'] = len(new_fp_df[new_fp_df['new_FP']==1])

    # Matrix de custo
    cost_matrix_dict = {
        'df_gearbox': {
            'Replacement_Cost': 100000,
            'Repair_Cost': 20000,
            'Inspection_cost': 5000
        },
        'df_generator': {
            'Replacement_Cost': 60000,
            'Repair_Cost': 15000,
            'Inspection_cost': 5000
        },
        'df_gen_bear': {
            'Replacement_Cost': 30000,
            'Repair_Cost': 12500,
            'Inspection_cost': 4500
        },
        'df_transformer': {
            'Replacement_Cost': 50000,
            'Repair_Cost': 3500,
            'Inspection_cost': 1500
        },
        'df_hydraulic': {
            'Replacement_Cost': 20000,
            'Repair_Cost': 3000,
            'Inspection_cost': 2000
        }
    }

    #Formula das poupancas
    Savings = 0
    fp_costs = cost_matrix_dict[component]['Inspection_cost'] * cf_numbers['FP']
    fn_costs = cost_matrix_dict[component]['Replacement_Cost'] * cf_numbers['FN']

    if len(met_cre_df[met_cre_df.TP == 1]) != 0:
        for i in range(len(dias_primeiro_TP)):
            tp_savings = cost_matrix_dict[component]['Replacement_Cost'] - (cost_matrix_dict[component]['Repair_Cost'] + (cost_matrix_dict[component]['Replacement_Cost'] - cost_matrix_dict[component]['Repair_Cost']) *(1 - (dias_primeiro_TP[i] / 60)))
    else:
        tp_savings = 0
    Savings = tp_savings - fp_costs - fn_costs

    return Savings, cf_numbers, met_cre_df



def scaler_antigo(df_train, df_test, scaler='StandardScaler'):
    for m_id in pd.unique(df_train.Turbine_ID):
        X_train = df_train.drop(columns=['Turbine_ID'])
        X_test = df_test.drop(columns=['Turbine_ID'])
        if scaler == 'MinMaxScaler':
            sc = MinMaxScaler()
            X_train_scale = sc.fit_transform(df)
            X_test_scale = sc.transform(X_test)
        else:
            sc = StandardScaler()
            X_train_scale = sc.fit_transform(X_train)
            X_test_scale = sc.transform(X_test)
    return X_train_scale, X_test_scale
