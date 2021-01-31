import pandas as pd
import numpy as np
import datetime

import Funtion_tools as Ft

#Esta função recebe como input o dataframe original das failures e devolve
# as variaveis alvo para todos os time stamps possíveis

def construct_df_target(df_failures):
    turbine_list=['T01','T06','T07', 'T09','T11']

    df_time_stamp=Ft.complete_time_df(Ft.time_df(),turbine_list)
    df_time_stamp['Timestamp'] = pd.to_datetime(df_time_stamp['Timestamp'])

    df_failures['Timestamp'] = pd.to_datetime(df_failures['Timestamp']).dt.tz_convert(None)
    df_failures['Timestamp_new'] = df_failures.apply(lambda x: x['Timestamp'] - datetime.timedelta(minutes=x['Timestamp'].minute % -10,
                             seconds=x['Timestamp'].second,
                             microseconds=x['Timestamp'].microsecond),axis=1)

    df_failures_stamped = pd.merge(df_time_stamp, df_failures , how='left', left_on=['Turbine_ID','Timestamp'], right_on=['Turbine_ID','Timestamp_new'])
    components_list = ['GEARBOX', 'GENERATOR', 'GENERATOR_BEARING', 'TRANSFORMER', 'HYDRAULIC_GROUP']

    for component in components_list:
        df_failures_stamped[f'Fail_{component}'] = df_failures_stamped.apply(lambda x: 1 if x['Component'] == component else 0, axis=1)

    df_failures_final=df_failures_stamped[['Timestamp_x','Turbine_ID','Fail_GEARBOX','Fail_GENERATOR','Fail_GENERATOR_BEARING','Fail_TRANSFORMER','Fail_HYDRAULIC_GROUP']].rename(columns = {'Timestamp_x': 'Timestamp'}, inplace = False)

    return df_failures_final
