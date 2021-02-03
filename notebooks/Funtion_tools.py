"Implementação de uma classe que disponibilizará funções para podermos usar de modo iterativo ao longo do projecto"

import os
import pandas as pd
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

def time_transform(df, time_column='Timestamp'):
    'Transformar as colunas referentes a tempo no data typw tempo'
    df[time_column] = pd.to_datetime(df[time_column].str[:19])
    # df[time_column] = df[time_column].dt.tz_localize('GMT')
    # df[time_column] = df[time_column].dt.tz_convert(None)
    return df

def time_intervals_max(df, time_column='Timestamp'):
    'Verificar o intervalo máximo de tempo que existe entre registos de tempo na mesma coluna'
    time_delta(df, time_column)
    return df['delta'].max()

def time_delta(df, time_column='Timestamp'):
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

def time_complete_df(df, turbine_list='T01'):
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

def graf_linhas(x, y):
    'Gráfico para variáveis continuas'
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(16,9))
    # Plot the responses for different events and regions
    sns.lineplot(x=x, y=y)
    return

def graf_barras(x, y):
    'Gráfico de barras para variáveis categóricas - Necessário fazer Dataframe de resumo'
    plt.figure(figsize=(16,4))
    sns.barplot(x=x, y=y, palette="deep")
    return

def graf_scatter(x, y):
    'ScatterPlot'
    sns.set_theme(style="darkgrid")
    sns.scatterplot(x=x, y=y, palette='deep')
    return

def timestamp_round_down(df, time_column='Timestamp'):
    'Arredondar os intervalos de tempo para os 10 minutos anteriores'
    df[time_column] = df.apply(lambda x: x[time_column] - datetime.timedelta(minutes=x[time_column].minute % 10,seconds=x[time_column].second, microseconds=x[time_column].microsecond),axis=1)
    return df

def time_measure(df, time_column='Timestamp', time_measure='hour'):
    'Arredondar os intervalos de tempo a medida de tempo desejada (Dias ou minutos)'
    if time_measure == 'hour':
        df[time_column] = df.apply(lambda x: x[time_column] - datetime.timedelta(minutes=x[time_column].minute % 10,seconds=x[time_column].second, microseconds=x[time_column].microsecond),axis=1)
    return df

def add_features(df_in, rolling_win_size):
    """Add rolling average and rolling standard deviation for sensors readings using fixed rolling window size.
    Args:
            df_in (dataframe)     : The input dataframe to be proccessed (training or test)
            rolling_win_size (int): The window size, number of cycles for applying the rolling function
    Returns:
            dataframe: contains the input dataframe with additional rolling mean and std for each sensor
    """

    sensor_cols = []
    for i in df_in.keys()[2:]:
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

    return df_out

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

def fail_dummies(df):
    '''Uniformização da tabela de logs e transformação com get_dummies'''
    #     Colunas a manter
    fail_cols_manter = ['Timestamp', 'Turbine_ID', 'Component']
    df = df[fail_cols_manter]
    #     transformação de Get_Dummies
    df = pd.get_dummies(df, columns=['Component'])
    return df
