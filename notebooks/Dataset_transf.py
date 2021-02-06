"Ficheiro de tratamento dos datasets"

import os
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

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
    df_hydraulic = df[time_id + pair_hyd + pair_rot + pair_amb + pair_blds + pair_cont + pair_nac + pair_spin + pair_bus]
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
