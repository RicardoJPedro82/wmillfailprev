"Implementação de uma classe que disponibilizará funções para podermos usar de modo iterativo ao longo do projecto"

import os
import pandas as pd
import datetime

def get_data():
    data = {}
    'obter um dicinário com todos as tabelas dos ficheiros em dataframes'
    #Obter o caminho do ficheiros
    root_dir = os.path.abspath("..//..")
    #Fazer uma lista dos ficheiros na pasta
    csv_path = os.path.join(root_dir, 'wmillfailprev', 'rawdata')
    #Lista dos ficheiros a importar
    lista = os.listdir(csv_path)
    lista = lista[1:]
    #Obter os nomes dos dataframes e criar o dicionário
    for i in range(len(lista)):
        if len(lista[i]) < 22:
            df_name = lista[i][:(len(lista[i]) - 4)] + '_df'
            df_name = df_name.replace('-','_')
        else:
            df_name = lista[i][12:len(lista[i]) - 4] + '_df'
            df_name = df_name.replace('-','_')
        data[df_name] = pd.read_csv(os.path.join(csv_path, lista[i]), sep=';')
    return  data

def power_curve_trans(Dict_df):
    df = Dict_df['Power_curve_df'].copy()
    df = pd.DataFrame(df['Wind speed (m/s),Power (kW)'].str.split(',',1).tolist(),columns = ['windspeed(m_s)','power(kw)'])
    df['windspeed(m_s)'] = df['windspeed(m_s)'].astype('float')
    df['power(kw)'] = df['power(kw)'].astype('int')
    df['power(kw)'].replace(to_replace=2, value=2000, inplace=True)
    return df

def logs_cols_uniform(df):
    'A partir de uma lista de draframe uniformizar os nomes relacionados com tempo e turbine_ID'
    df = df.rename(columns={'TimeDetected': 'Timestamp', 'UnitTitle':'Turbine_ID'})
    return df

def transform_time(df, time_column):
    'Transformar as colunas referentes a tempo no data typw tempo'
    df[time_column] = pd.to_datetime(df[time_column])
    return df

def max_time_intervals(df, time_column):
    'Verificar o intervalo máximo de tempo que existe entre registos de tempo na mesma coluna'
    time_delta(df, time_column)
    return df['delta'].max()

def time_delta(df, time_column):
    'Adicionar uma coluna de diferenças entre linhas de tempo'
    transform_time(df, time_column)
    df['delta'] = (df[time_column]-df[time_column].shift()).fillna(0)
    return df

def time_df(ano=2016, mes=1, dia=1):
    '''Criar um dataframe a partir de uma data inicial para os próximos dois 2 de 10 em 10 minutos'''
    time_list = []
    # Criar o dataframe de tempo
    for i in range((365*2*24*6)+1):
        time_list.append(pd.datetime(ano,mes,dia)+pd.Timedelta(minutes=10)*i)
    time_df = pd.DataFrame(time_list, columns={'Timestamp'})
    #Criação de Dataframe vazio para fazer append
    time_df = time_df.dropna().reset_index(drop='index')
    return time_df

def complete_time_df(df, turbine_list):
    # Multiplicar o intervalo de tempo pelas turbinas
    df_vazio = pd.DataFrame(index=[0,1], columns=['Timestamp', 'Turbine_ID'])
    for i in turbine_list:
        passage = df.copy()
        passage['Turbine_ID'] = i
        df_vazio = pd.concat([df_vazio, passage], sort=True)
    # Corrigir o Dataframe
    df = df_vazio.dropna().reset_index(drop='index')
    return df

def merge_df(df1, df2, date_until=None, date_after=None):
    'Colocar apenas uma data no formato AAAA-MM-DD HH:MM:SS - fazer o merge dos dataframes respeitando as datas escolhidas para os ficheiros de     teste e treino'
    df = pd.merge(df1, df2, how='left', on=['Timestamp', 'Turbine_ID'])
    # Criar a lista de campo do df_merged
    campos_col = []
    for i in df.keys():
        campos_col.append(i)
    # fazer o slice do df_merged
    if date_until != None and date_after == None:
        df = df.loc[df['Timestamp'] < date_until, campos_col]
    elif date_until == None and date_after != None:
        df = df.loc[df['Timestamp'] >= date_after, campos_col]
    elif date_until != None and date_after != None:
        return 'Escolher apenas uma das duas datas possiveis'
    return df
