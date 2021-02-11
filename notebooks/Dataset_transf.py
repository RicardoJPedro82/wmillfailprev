"Ficheiro de tratamento dos datasets"

import os
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import Dataset_transf as dprep
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def get_data():
    data = {}
    'obter um dicinário com todos as tabelas dos ficheiros em dataframes'
    #Obter o caminho do ficheiros
    root_dir = os.path.abspath("..//..")
    #Fazer uma lista dos ficheiros na pasta
    csv_path = os.path.join(root_dir, 'wmillfailprev', 'rawdata')
    # Importar os datasets
    # Failures
    failures_df = pd.read_csv(os.path.join(csv_path, 'wind-farm-1-failures-training.csv'), sep=';')
    # Logs de comentários
    logs_df = pd.read_csv(os.path.join(csv_path, 'wind-farm-1-logs-training.csv'), sep=';')
        # Uniformizar os cabaçalhos das colunas
    logs_df = logs_cols_uniform(logs_df)
    logs_df = logs_df[['Timestamp', 'Turbine_ID', 'Remark']]
    # Sinais scada
    signals_df = pd.read_csv(os.path.join(csv_path, 'wind-farm-1-signals-training.csv'), sep=';')

    # alteração do Timestamp para tipo date
    failures_df = time_transform(failures_df)
    signals_df = time_transform(signals_df)
    logs_df = time_transform(logs_df)

    # uniformizaçao dos intervalos de tempo
    failures_df = timestamp_round_down(failures_df)
    logs_df = timestamp_round_down(logs_df)
    signals_df = timestamp_round_down(signals_df)

    # Columns Drop of unrelated features
    drop_features_signals = ['Prod_LatestAvg_ActPwrGen2', 'Prod_LatestAvg_ReactPwrGen2']
    signals_df = signals_df.drop(columns=drop_features_signals)
    failures_df = failures_df.drop(columns='Remarks')

    # fill nans
    signals_df = signals_df.bfill()

    return failures_df, signals_df, logs_df

def logs_cols_uniform(df):
    'A partir de uma lista de draframe uniformizar os nomes relacionados com tempo e turbine_ID'
    df = df.rename(columns={'TimeDetected': 'Timestamp', 'UnitTitle':'Turbine_ID'})
    return df

def timestamp_round_down(df, time_column='Timestamp'):
    'Arredondar os intervalos de tempo para os 10 minutos anteriores'
    df[time_column] = df.apply(lambda x: x[time_column] - datetime.timedelta(minutes=x[time_column].minute % 10,seconds=x[time_column].second, microseconds=x[time_column].microsecond),axis=1)
    return df

def time_transform(df, time_column='Timestamp'):
    'Transformar as colunas referentes a tempo no data type tempo'
    df[time_column] = pd.to_datetime(df[time_column]).dt.tz_convert(None)
    # df[time_column] = df[time_column]
    return df

def component(component, col):
    pair_comp_col=[]
    for i in col:
        if component in i:
            pair_comp_col.append(i)
    return pair_comp_col

def component_df_creation(df):
    # Retornar dataframes por tipo de componente
    time_id = ['Timestamp', 'Turbine_ID']
    pair_hyd = component('Hyd', df.columns)
    pair_trafo = component('Trafo', df.columns)
    pair_gear = component('Gear', df.columns)
    pair_gen = component('Gen', df.columns)
    pair_rot = component('Rtr', df.columns)
    pair_amb = component('Amb', df.columns)
    pair_blds = component('Blds', df.columns)
    pair_cont = component('Cont', df.columns)
    pair_nac = component('Nac', df.columns)
    pair_spin = component('Spin', df.columns)
    pair_bus = component('Busbar', df.columns)
    pair_vol = component('Volt', df.columns)

    #Create DF for each component
    df_generator = df[time_id + pair_gen + pair_rot + pair_amb + pair_blds + pair_cont + pair_nac + pair_spin + pair_bus + pair_hyd]
    df_gen_bear = df[time_id + pair_gen + pair_rot + pair_amb + pair_blds + pair_cont + pair_nac + pair_spin + pair_bus + pair_hyd]
    df_transformer = df[time_id + pair_trafo + pair_rot + pair_amb + pair_blds + pair_cont + pair_nac + pair_spin + pair_bus + pair_vol]
    df_hydraulic = df[time_id + pair_hyd + pair_rot + pair_amb + pair_blds + pair_cont + pair_nac + pair_spin + pair_bus + pair_vol]
    df_gearbox = df[time_id + pair_gear + pair_rot + pair_amb + pair_blds + pair_cont + pair_nac + pair_spin + pair_bus + pair_hyd]

    return df_generator, df_gen_bear, df_transformer, df_hydraulic, df_gearbox

def fail_dummies(df):
    '''Uniformização da tabela de logs e transformação com get_dummies'''
    #     Colunas a manter
    fail_cols_manter = ['Timestamp', 'Turbine_ID', 'Component']
    df = df[fail_cols_manter]
    #     transformação de Get_Dummies
    df = pd.get_dummies(df, columns=['Component'])
    return df

def logs_dummies(df):
    '''Uniformização da tabela de logs e transformação com get_dummies'''
    #     Colunas a manter
    logs_cols_manter = ['Timestamp', 'Turbine_ID', 'Remark']
    #     Lista de comentários a manter
    lista_comentarios = [
        'External power ref.:2000kW', 'Generator 1 in',
        'Hot HV trafo 270°C      0kW', 'Yaw Speed Exc:  0° Rate:10sec',
        'GearoilCooler 2, gear:  57°C', 'GearoilCooler 1, gear:  49°C',
        'Gen. int. vent. 1, temp: 50°C', 'Gen. ext. vent. 1, temp: 50°C',
        'Gen. int. vent. 0, temp: 34°C', 'Gen. ext. vent. 0, temp: 34°C',
        'Gen. ext. vent. 2, temp: 65°C', 'Gen. ext. vent. 1, temp: 49°C',
        'Yawcontr. from:30010 to:30011', 'Yawcontr. from:30011 to:30010',
        'Accumulator test done -> OK'
    ]
    #     Separação de comentários a manter
    df = df.loc[df['Remark'].isin(lista_comentarios), logs_cols_manter]
    #     transformação de Get_Dummies
    df = pd.get_dummies(df, columns=['Remark'])
    return df

def sig_fail_merge_dfs(sig_df, fail_df, component):
    'fazer o merge com o failures e desevolver o já dumminized'
    #filtrar o componente
    fail_df = fail_df[fail_df['Component'] == component]
    # aplicar o dummies
    fail_df = fail_dummies(fail_df)
    # fazer o merge
    df_merged = sig_df.merge(fail_df, on=['Timestamp','Turbine_ID'], how='outer')
    # colocar zeros
    df_merged.rename(columns= {'Component_' + component:'Component'}, inplace=True)
    df_merged['Component'] = df_merged['Component'].fillna(0)
    df_merged = df_merged.sort_values(by=['Turbine_ID','Timestamp'])
    df_merged.fillna(0, inplace=True)
    return df_merged

def fill_na_by_turbine(df, turbines_list):
    df_ = pd.DataFrame(columns=df.columns)
    for turbine in turbines_list:
        df1 = df.loc[df['Turbine_ID']==turbine]
        if df1['Component'].nunique()>1:
            index = df1[df1['Component']==1]
            index['date'] = index['Timestamp']
            index = index[['date','Timestamp', 'Turbine_ID']]
            df_merged = df1.merge(index, how='left', on=['Turbine_ID','Timestamp'])
            df_merged = df_merged.fillna(method='bfill')

            #If there is not a failure after, hold present date
            df_merged['date'] = df_merged['date'].fillna(df_merged['Timestamp'])

            df_merged['TTF'] = round((df_merged['date'] -
                                      df_merged['Timestamp']) / np.timedelta64(1, 'D'),0)
            # df_merged = df_merged.fillna(method='Bfill')
        else:
            df_merged = df1
            df_merged['date'] = df_merged['Timestamp']
            df_merged['TTF'] = 0 # df_merged['date'] - df_merged['Timestamp']
            # df_merged = df_merged.fillna(method='Bfill')
        #Drop Column Date
        df_final = df_merged.drop(columns='date')

        #df_final['TTF'] = df_final['TTF'].dt.days

        df_ = pd.concat([df_, df_final])

    return df_


def Failure_Time_Horizon(days, period):
    if 2 <= days <= period:
        Flag=1
    else:
        Flag=0
    return Flag

def aplic_var_target(df, period):
    nome = str(period)
    nome = nome+'_days'
    df[nome] = df.apply(lambda x: Failure_Time_Horizon(x['TTF'], period),axis=1)
    return df

def prepare_train_df(df, meses=3):
    if 'Timestamp' in df.keys():
        last_date = df['Timestamp'].iloc[-1]
        split = last_date - pd.DateOffset(months=meses)
        df_train = df[df['Timestamp'] < split]
    else:
        last_date = df['Date'].iloc[-1]
        split = last_date - pd.DateOffset(months=meses)
        df_train = df[df['Date'] < split]

    # df_test = df[df['Timestamp'] >= split]
    return df_train

def prepare_test_df(df, meses=3):
    if 'Timestamp' in df.keys():
        last_date = df['Timestamp'].iloc[-1]
        split = last_date - pd.DateOffset(months=meses)
        df_test = df[df['Timestamp'] >= split]
    else:
        last_date = df['Date'].iloc[-1]
        split = last_date - pd.DateOffset(months=meses)
        df_test = df[df['Date'] >= split]
    return df_test

def add_features(df_in, rolling_win_size=15):
    """Add rolling average and rolling standard deviation for sensors readings using fixed rolling window size.
    Args:
            df_in (dataframe)     : The input dataframe to be proccessed (training or test)
            rolling_win_size (int): The window size, number of cycles for applying the rolling function
    Returns:
            dataframe: contains the input dataframe with additional rolling mean and std for each sensor
    """

    sensor_cols = []
    index = df_in.columns.get_loc('TTF')
    for i in df_in.columns[2:index]:
        sensor_cols.append(i)

    sensor_av_cols = [nm+'_av' for nm in sensor_cols]
    sensor_sd_cols = [nm+'_sd' for nm in sensor_cols]

    df_out = pd.DataFrame()

    ws = rolling_win_size

    #calculate rolling stats for each engine id

    for m_id in pd.unique(df_in.Turbine_ID):

        # get a subset for each engine sensors
        df_engine = df_in[df_in['Turbine_ID'] == m_id]
        df_sub = df_engine[sensor_cols]

        # get rolling mean for the subset
        av = df_sub.rolling(ws, min_periods=1).mean()
        av.columns = sensor_av_cols

        # get the rolling standard deviation for the subset
        sd = df_sub.rolling(ws, min_periods=1).std().fillna(0)
        sd.columns = sensor_sd_cols

        # combine the two new subset dataframes columns to the engine subset
        new_ftrs = pd.concat([df_engine,av,sd], axis=1)

        # add the new features rows to the output dataframe
        df_out = pd.concat([df_out,new_ftrs])
    df_out = df_out.sort_values(by=['Turbine_ID', 'Date']   )
    return df_out

def group_por_frequency(df, period='Dia', strategy='mean'):
    'Função para agregar o data-frame pela medida de tempo pretendida, periodo _Dia_ ou _Hora_'
    if period == 'Dia':
        df['Date'] = df['Timestamp'].dt.date
    elif period == 'Hora':
        df['Date'] = df.apply(lambda x: x['Timestamp'] - datetime.timedelta(hours=x['Timestamp'].hour % -1, minutes=x['Timestamp'].minute, seconds=x['Timestamp'].second, microseconds=x['Timestamp'].microsecond),axis=1)
    else:
        print('Medida de tempo não suportada')

    if strategy == 'max':
        df = df.groupby(by=['Turbine_ID','Date']).max().reset_index().drop(columns='Timestamp')
    else:
        df = df.groupby(by=['Turbine_ID','Date']).mean().reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

#Standard scaler per Turbine
def scale(df_train, df_test, scaler='StandardScaler'):

    X_train = df_train
    X_test = df_test

    X_train1 = X_train.loc[X_train['Turbine_ID']=='T01']
    X_test1 = X_test.loc[X_test['Turbine_ID']=='T01']

    X_train1 = X_train1.drop(columns='Turbine_ID')
    X_test1 = X_test1.drop(columns='Turbine_ID')

    if scaler == 'MinMaxScaler':
        sc = MinMaxScaler()
        X_train1 = sc.fit_transform(X_train1)
        X_test1 = sc.transform(X_test1)
    else:
        sc = StandardScaler()
        X_train1 = sc.fit_transform(X_train1)
        X_test1 = sc.transform(X_test1)

    turbines = ['T06', 'T07', 'T09', 'T11']
    for turbine in turbines:
        X_train_ = X_train.loc[X_train['Turbine_ID']==turbine]
        X_test_ = X_test.loc[X_test['Turbine_ID']==turbine]

        X_train_ = X_train_.drop(columns='Turbine_ID')
        X_test_ = X_test_.drop(columns='Turbine_ID')

        if scaler == 'MinMaxScaler':
            sc = MinMaxScaler()
            X_train_ = sc.fit_transform(X_train_)
            X_test_ = sc.transform(X_test_)
        else:
            sc = StandardScaler()
            X_train_ = sc.fit_transform(X_train_)
            X_test_ = sc.transform(X_test_)

        X_train1 = np.concatenate((X_train1, X_train_))
        X_test1 = np.concatenate((X_test1, X_test_))

    return X_train1, X_test1

def bin_classify(model, clf, X_train, X_test, y_train, y_test, params=None, score=None, ):
    """Perform Grid Search hyper parameter tuning on a classifier.
    Args:
        model (str): The model name identifier
        clf (clssifier object): The classifier to be tuned
        features (list): The set of input features names
        params (dict): Grid Search parameters
        score (str): Grid Search score
    Returns:
        Tuned Clssifier object
        dataframe of model predictions and scores
    """

    grid_search = model_selection.GridSearchCV(estimator=clf, param_grid=params, cv=5, scoring=score)

    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)

    if hasattr(grid_search, 'predict_proba'):
        y_score = grid_search.predict_proba(X_test)[:,1]
    elif hasattr(grid_search, 'decision_function'):
        y_score = grid_search.decision_function(X_test)
    else:
        y_score = y_pred

    predictions = {'y_pred' : y_pred, 'y_score' : y_score}
    df_predictions = pd.DataFrame.from_dict(predictions)

    return grid_search.best_estimator_, df_predictions

def bin_class_metrics(model, y_test, y_pred, y_score, print_out=True, plot_out=True):
    """Calculate main binary classifcation metrics, plot AUC ROC and Precision-Recall curves.
    Args:
        model (str): The model name identifier
        y_test (series): Contains the test label values
        y_pred (series): Contains the predicted values
        y_score (series): Contains the predicted scores
        print_out (bool): Print the classification metrics and thresholds values
        plot_out (bool): Plot AUC ROC, Precision-Recall, and Threshold curves
    Returns:
        dataframe: The combined metrics in single dataframe
        dataframe: ROC thresholds
        dataframe: Precision-Recall thresholds
        Plot: AUC ROC
        plot: Precision-Recall
        plot: Precision-Recall threshold; also show the number of engines predicted for maintenace per period (queue).
        plot: TPR-FPR threshold
    """

    binclass_metrics = {
                        'Accuracy' : metrics.accuracy_score(y_test, y_pred),
                        'Precision' : metrics.precision_score(y_test, y_pred),
                        'Recall' : metrics.recall_score(y_test, y_pred),
                        'F1 Score' : metrics.f1_score(y_test, y_pred),
                        'ROC AUC' : metrics.roc_auc_score(y_test, y_score)
                       }

    df_metrics = pd.DataFrame.from_dict(binclass_metrics, orient='index')
    df_metrics.columns = [model]


    fpr, tpr, thresh_roc = metrics.roc_curve(y_test, y_score)

    roc_auc = metrics.auc(fpr, tpr)

    engines_roc = []
    for thr in thresh_roc:
        engines_roc.append((y_score >= thr).mean())

    engines_roc = np.array(engines_roc)

    roc_thresh = {
                    'Threshold' : thresh_roc,
                    'TPR' : tpr,
                    'FPR' : fpr,
                    'Que' : engines_roc
                 }

    df_roc_thresh = pd.DataFrame.from_dict(roc_thresh)

    #calculate other classification metrics: TP, FP, TN, FN, TNR, FNR
    #from ground truth file, positive class = 25 => TP + FN = 25
    #from ground truth file, negative class = 75 => TN + FP = 75

    df_roc_thresh['TP'] = (25*df_roc_thresh.TPR).astype(int)
    df_roc_thresh['FP'] = (25 - (25*df_roc_thresh.TPR)).astype(int)
    df_roc_thresh['TN'] = (75*(1 - df_roc_thresh.FPR)).astype(int)
    df_roc_thresh['FN'] = (75 - (75*(1 - df_roc_thresh.FPR))).astype(int)

    df_roc_thresh['TNR'] = df_roc_thresh['TN']/(df_roc_thresh['TN'] + df_roc_thresh['FN'])
    df_roc_thresh['FNR'] = df_roc_thresh['TN']/(df_roc_thresh['TN'] + df_roc_thresh['FP'])

    df_roc_thresh['Model'] = model



    precision, recall, thresh_prc = metrics.precision_recall_curve(y_test, y_score)

    thresh_prc = np.append(thresh_prc,1)

    engines_prc = []
    for thr in thresh_prc:
        engines_prc.append((y_score >= thr).mean())

    engines_prc = np.array(engines_prc)

    prc_thresh = {
                    'Threshold' : thresh_prc,
                    'Precision' : precision,
                    'Recall' : recall,
                    'Que' : engines_prc
                 }

    df_prc_thresh = pd.DataFrame.from_dict(prc_thresh)
    cf_matrix = metrics.confusion_matrix(y_test, y_pred)

    # if print_out:
    #     print('-----------------------------------------------------------')
    #     print(model, '\n')
    #     print('Confusion Matrix:')
    #     print(cf_matrix)
    #     print('\nClassification Report:')
    #     print(metrics.classification_report(y_test, y_pred))
    #     print('\nMetrics:')
    #     print(df_metrics)

    #     print('\nROC Thresholds:\n')
    #     print(df_roc_thresh[['Threshold', 'TP', 'FP', 'TN', 'FN', 'TPR', 'FPR', 'TNR','FNR', 'Que']])

    #     print('\nPrecision-Recall Thresholds:\n')
    #     print(df_prc_thresh[['Threshold', 'Precision', 'Recall', 'Que']])

    if plot_out:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False )
        fig.set_size_inches(10,10)

        ax1.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f'% roc_auc)
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([-0.05, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(loc="lower right", fontsize='small')

        ax2.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend(loc="lower left", fontsize='small')

        ax3.plot(thresh_roc, fpr, color='red', lw=2, label='FPR')
        ax3.plot(thresh_roc, tpr, color='green',label='TPR')
        ax3.plot(thresh_roc, engines_roc, color='blue',label='Engines')
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('%')
        ax3.legend(loc='upper right', fontsize='small')

        ax4.plot(thresh_prc, precision, color='red', lw=2, label='Precision')
        ax4.plot(thresh_prc, recall, color='green',label='Recall')
        ax4.plot(thresh_prc, engines_prc, color='blue',label='Engines')
        ax4.set_ylim([0.0, 1.05])
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('%')
        ax4.legend(loc='lower left', fontsize='small')

    return  df_metrics, df_roc_thresh, df_prc_thresh, cf_matrix

'A partir daqui são funções para criar a métrica de avaliação'

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

def metrics_create_df(df_test_in, y_test_in, y_pred_in):
    'Criar o dataframe de avaliação dos resultados da predição'
    cols= ['Date','Turbine_ID', 'TTF']
    met_cre_df = df_test_in[cols].copy()
    met_cre_df['y_test'] = y_test_in
    met_cre_df['y_pred'] = y_pred_in
    met_cre_df = met_cre_df.reset_index().drop(columns='index')
    met_cre_df['TP'] = [met_poupanca_TP(met_cre_df.y_test[i], met_cre_df.y_pred[i]) for i in range(len(met_cre_df.y_pred))]
    met_cre_df['TN'] = [met_poupanca_TN(met_cre_df.y_test[i], met_cre_df.y_pred[i]) for i in range(len(met_cre_df.y_pred))]
    met_cre_df['FP'] = [met_poupanca_FP(met_cre_df.y_test[i], met_cre_df.y_pred[i]) for i in range(len(met_cre_df.y_pred))]
    met_cre_df['FN'] = [met_poupanca_FN(met_cre_df.y_test[i], met_cre_df.y_pred[i]) for i in range(len(met_cre_df.y_pred))]

    #'Dataframe com as falhas'
    falhas = met_cre_df[met_cre_df.TTF.between(0.1 , 1.0)]

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
            cf_numbers['TN'] = 0
            cf_numbers['FP'] = 0
            cf_numbers['FN'] = 0
        elif cf_numbers['FN'] >= 1:
            cf_numbers['TP'] = 0
            cf_numbers['TN'] = 0
            cf_numbers['FP'] = 0
            cf_numbers['FN'] = 1
        dias_primeiro_TP.append(ocorrencias.loc[ocorrencias[ocorrencias.TP == 1].index[0]].TTF.astype(int))

    # Matrix de custo
    cost_matrix_dict = {
        'GEARBOX': {
            'Replacement_Cost': 100000,
            'Repair_Cost': 20000,
            'Inspection_cost': 5000
        },
        'GENERATOR': {
            'Replacement_Cost': 60000,
            'Repair_Cost': 15000,
            'Inspection_cost': 5000
        },
        'GENERATOR_BEARING': {
            'Replacement_Cost': 30000,
            'Repair_Cost': 12500,
            'Inspection_cost': 4500
        },
        'TRANSFORMER': {
            'Replacement_Cost': 50000,
            'Repair_Cost': 3500,
            'Inspection_cost': 1500
        },
        'HYDRAULIC_GROUP': {
            'Replacement_Cost': 20000,
            'Repair_Cost': 3000,
            'Inspection_cost': 2000
        }
    }

    #Formula das poupancas
    Savings = 0
    for i in range(len(dias_primeiro_TP)):
    Savings = Savings -cost_matrix_dict['GENERATOR']['Inspection_cost'] * cf_numbers['FP'] - cost_matrix_dict['GENERATOR']['Replacement_Cost'] * cf_numbers['FN'] + cost_matrix_dict['GENERATOR']['Replacement_Cost'] - (cost_matrix_dict['GENERATOR']['Repair_Cost'] + (cost_matrix_dict['GENERATOR']['Replacement_Cost'] - cost_matrix_dict['GENERATOR']['Repair_Cost']) *(1 - (dias_primeiro_TP[i] / 60)))

    return Savings
