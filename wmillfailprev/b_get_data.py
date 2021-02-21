# -*- coding: UTF-8 -*-
""" Main lib for wmillfailprev Project
"""

import os
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, roc_curve, precision_score, recall_score,confusion_matrix,f1_score,fbeta_score, make_scorer

pd.set_option('display.width', 200)

def get_data(file):
    'file = full path of the original file. obter os dataframes dos ficheiros'
    df = pd.read_csv(file, sep=';')
    df = time_transform(df)
    df = timestamp_round_down(df)
    df = df.bfill()
    return df

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

def sig_fail_merge_dfs(sig_df, fail_df, component):
    'fazer o merge com o failures e desevolver o já dummyfied'
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
    df_ = pd.DataFrame(columns=df.columns, dtype='int64')
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

            df_merged['TTF'] = round((df_merged['date'] - df_merged['Timestamp']) / np.timedelta64(1, 'D'),0)
            df_merged = df_merged.fillna(method='Bfill')
        else:
            df_merged = df1
            df_merged['date'] = df_merged['Timestamp']
            df_merged['TTF'] = 0 # df_merged['date'] - df_merged['Timestamp']
            # df_merged = df_merged.fillna(method='Bfill')
        #Drop Column Date
        df_final = df_merged.drop(columns='date')

        #df_final['TTF'] = df_final['TTF'].dt.days

        df_ = pd.concat([df_, df_final])
        df_['Timestamp'] = pd.to_datetime(df_['Timestamp'])
    return df_

def fill_na_by_turb_predict(df, turbines_list):
    df = df.fillna(method='bfill')
    return df

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

def add_features(df_in, rolling_win_size=15):
    """Add rolling average and rolling standard deviation for sensors readings using fixed rolling window size.
    """
    cols =['Turbine_ID', 'Date', 'TTF', '60_days', 'Component']
    other_cols = []
    for i in df_in.columns:
        if i not in cols:
            other_cols.append(i)
    all_cols = cols + other_cols

    df_in = df_in[all_cols]

    sensor_cols = []
    for i in df_in.columns[5:]:
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

def add_feat_predict(df_in, rolling_win_size=15):
    """Add rolling average and rolling standard deviation for sensors readings using fixed rolling window size.
    """
    cols =['Turbine_ID', 'Date']
    other_cols = []
    for i in df_in.columns:
        if i not in cols:
            other_cols.append(i)
    all_cols = cols + other_cols

    df_in = df_in[all_cols]

    sensor_cols = []
    for i in df_in.columns[2:]:
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

if __name__ == "__main__":
    print('001 - Obtendo os dados')
    # Obter o caminho dos ficheiros.
    root_dir = os.path.abspath('..')
    csv_path = os.path.join(root_dir, 'rawdata')
    # Importar o dataset de failures
    failures_path = os.path.join(csv_path, 'wind-farm-1-failures-training.csv')
    failures_df = get_data(failures_path)
    # Importar o dataset de signals
    signals_path = os.path.join(csv_path, 'wind-farm-1-signals-training.csv')
    signals_df = get_data(signals_path)
    # Cortar colunas que não têm valores
    cols_to_drop = ['Prod_LatestAvg_ActPwrGen2', 'Prod_LatestAvg_ReactPwrGen2']
    signals_df = signals_df.drop(columns=cols_to_drop)
    print('002 - Criar o dicionário com os Dataframes originais')
    df_dict = {'failures_df':failures_df, 'signals_df':signals_df}
    print('003 - Criar os datasets por componentes')
    df_generator, df_gen_bear, df_transformer, df_hydraulic, df_gearbox = component_df_creation(signals_df)
    print('004 - Criar o dicionário de datasets por componentes')
    comp_df_dict = {'df_generator': df_generator,'df_hydraulic': df_hydraulic,'df_gen_bear': df_gen_bear,'df_transformer': df_transformer,'df_gearbox': df_gearbox}
    print('005 - fazer cópia dos dataframes')
    comp_prep_df_dict = comp_df_dict.copy()
    print('006 - Merge com o dataframe de falhas')
    component_list = ['GENERATOR', 'HYDRAULIC_GROUP', 'GENERATOR_BEARING', 'TRANSFORMER','GEARBOX']
    for i, key in enumerate(comp_prep_df_dict):
        comp_prep_df_dict[key] = sig_fail_merge_dfs(sig_df=comp_prep_df_dict[key],fail_df=failures_df,component=component_list[i])
    print('007 - Fillna by turbine')
    turbine_list = ['T11', 'T06', 'T01', 'T09', 'T07']
    for i, key in enumerate(comp_prep_df_dict):
        comp_prep_df_dict[key] = fill_na_by_turbine(comp_prep_df_dict[key],turbine_list)
    print('008 - Criação da variável alvo')
    for key in comp_prep_df_dict:
        comp_prep_df_dict[key] = aplic_var_target(comp_prep_df_dict[key], 60)
    print('009 - retirar as colunas que não se relacionam com a variavel alvo')
    for i, key in enumerate(comp_prep_df_dict):
        comp_prep_df_dict[key] = comp_prep_df_dict[key].drop(columns=feat_drop_list[i])
    print('010 - agrupar pela medida de tempo seleccionada')
    for key in comp_prep_df_dict:
        comp_prep_df_dict[key] = group_por_frequency(comp_prep_df_dict[key], period='Dia')
    print('011 - Adicionar medidas de alisamento')
    for key in comp_prep_df_dict:
        comp_prep_df_dict[key] = add_features(comp_prep_df_dict[key], rolling_win_size=10)
        print(comp_prep_df_dict[key].shape)
