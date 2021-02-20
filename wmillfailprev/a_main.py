from b_get_data import *
from c_model_related import CustomStandardScaler as Cscl
from c_model_related import Trainer as tr

import joblib

gen_features_drop = ['Gen_RPM_Max', 'Gen_RPM_Min', 'Gen_Phase1_Temp_Avg', 'Gen_Phase3_Temp_Avg','Amb_WindSpeed_Est_Avg', 'Grd_RtrInvPhase1_Temp_Avg','Grd_RtrInvPhase3_Temp_Avg', 'Rtr_RPM_Max', 'Rtr_RPM_Min','Blds_PitchAngle_Max', 'Blds_PitchAngle_Min','Prod_LatestAvg_ReactPwrGen1', 'Cont_Hub_Temp_Avg', 'Spin_Temp_Avg','Rtr_RPM_Std', 'Rtr_RPM_Avg', 'Cont_VCP_Temp_Avg']

gen_bear_features_drop = ['Gen_RPM_Max', 'Gen_RPM_Min', 'Gen_Phase1_Temp_Avg', 'Gen_Phase3_Temp_Avg','Amb_WindSpeed_Est_Avg', 'Grd_RtrInvPhase1_Temp_Avg','Grd_RtrInvPhase3_Temp_Avg', 'Rtr_RPM_Max', 'Rtr_RPM_Min','Blds_PitchAngle_Max', 'Blds_PitchAngle_Min','Prod_LatestAvg_ReactPwrGen1', 'Cont_Hub_Temp_Avg', 'Spin_Temp_Avg','Rtr_RPM_Std', 'Rtr_RPM_Avg', 'Cont_VCP_Temp_Avg']

hyd_features_drop = ['Rtr_RPM_Max', 'Rtr_RPM_Min', 'Blds_PitchAngle_Max', 'Blds_PitchAngle_Min','Blds_PitchAngle_Max', 'Grd_RtrInvPhase3_Temp_Avg', 'Grd_Busbar_Temp_Avg','Amb_WindSpeed_Est_Avg', 'Spin_Temp_Avg', 'Cont_Hub_Temp_Avg','Grd_RtrInvPhase1_Temp_Avg', 'Cont_VCP_Temp_Avg']

gearbox_features_drop = ['Rtr_RPM_Max', 'Rtr_RPM_Min', 'Grd_RtrInvPhase1_Temp_Avg','Grd_RtrInvPhase3_Temp_Avg', 'Blds_PitchAngle_Min', 'Blds_PitchAngle_Max','Cont_VCP_Temp_Avg', 'Grd_Busbar_Temp_Avg', 'Amb_WindSpeed_Est_Avg','Spin_Temp_Avg']

transf_features_drop = ['HVTrafo_Phase1_Temp_Avg', 'HVTrafo_Phase3_Temp_Avg', 'Rtr_RPM_Max','Rtr_RPM_Min', 'Grd_RtrInvPhase1_Temp_Avg', 'Grd_RtrInvPhase3_Temp_Avg','Blds_PitchAngle_Min', 'Blds_PitchAngle_Max', 'Amb_WindSpeed_Est_Avg','Spin_Temp_Avg', 'Cont_VCP_Temp_Avg']

feat_drop_list = [gen_features_drop, hyd_features_drop, gen_bear_features_drop,transf_features_drop, gearbox_features_drop]

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

    print('011 - Separar entre treino e teste')
    df_train_comp_dict = {}
    df_test_comp_dict = {}
    for key in comp_prep_df_dict:
        df_train_comp_dict[key] = prepare_train_df(comp_prep_df_dict[key], meses = 3)
        df_test_comp_dict[key] = prepare_test_df(comp_prep_df_dict[key], meses = 3)

    print('012 - separar entre x_train, x_test')
    x_train = df_train_comp_dict.copy()
    x_test = df_test_comp_dict.copy()
    y_train = df_train_comp_dict.copy()
    y_test = df_test_comp_dict.copy()
    # retirar as colunas que não devem entrar no x

    print('012.1 - considerar no y set apenas as turbinas que tiveram falhas')
    train_turbines = {'df_generator': ['T11', 'T06'],'df_hydraulic': ['T06', 'T11'],'df_gen_bear': ['T07', 'T09'],'df_transformer': ['T07'],'df_gearbox': ['T09']}
    for key in y_train:
        y_train[key] = y_train[key][y_train[key]['Turbine_ID'].isin(train_turbines[key])]
    col_to_mantain_test = ['60_days']
    for key in y_train:
        y_train[key] = y_train[key][col_to_mantain_test].to_numpy().astype(int)[:,0]
        y_test[key] = y_test[key][col_to_mantain_test].to_numpy().astype(int)[:,0]

    print('013 - Adicionar medidas de alisamento')
    for key in x_train:
        x_train[key] = add_features(x_train[key], rolling_win_size=10)
        x_test[key] = add_features(x_test[key], rolling_win_size=10)

    print('012 - separar entre x_train, x_test')
    # retirar as colunas que não devem entrar no x
    cols_to_drop_train = ['Date', '60_days', '60_days_av', '60_days_sd']
    for key in x_train:
        x_train[key] = x_train[key].drop(columns=cols_to_drop_train)
        x_test[key] = x_test[key].drop(columns=cols_to_drop_train)

    print('014 - fazer o fit do CustomTransfMiguel para cada componente à pedreiro')
    print('014.1 - df_generator')
    scaler_df_generator = Cscl()
    scaler_df_generator.fit(x_train['df_generator'])
    joblib.dump(scaler_df_generator, 'scaler_df_generator.joblib')
    print('014.2 - df_hydraulic')
    scaler_df_hydraulic = Cscl()
    scaler_df_hydraulic.fit(x_train['df_hydraulic'])
    joblib.dump(scaler_df_hydraulic, 'scaler_df_hydraulic.joblib')
    print('014.3 - df_gen_bear')
    scaler_df_gen_bear = Cscl()
    scaler_df_gen_bear.fit(x_train['df_gen_bear'])
    joblib.dump(scaler_df_gen_bear, 'scaler_df_gen_bear.joblib')
    print('014.4 - df_transformer')
    scaler_df_transformer = Cscl()
    scaler_df_transformer.fit(x_train['df_transformer'])
    joblib.dump(scaler_df_transformer, 'scaler_df_transformer.joblib')
    print('014.5 - df_gearbox')
    scaler_df_gearbox = Cscl()
    scaler_df_gearbox.fit(x_train['df_gearbox'])
    joblib.dump(scaler_df_gearbox, 'scaler_df_gearbox.joblib')

    print('015 - considerar no train set apenas as turbinas que tiveram falhas')
    train_turbines = {'df_generator': ['T11', 'T06'],'df_hydraulic': ['T06', 'T11'],'df_gen_bear': ['T07', 'T09'],'df_transformer': ['T07'],'df_gearbox': ['T09']}
    for key in x_train:
        x_train[key] = x_train[key][x_train[key]['Turbine_ID'].isin(train_turbines[key])]

    print('016 - aplicar o scale com o fit efectuado no treino')
    x_train['df_generator'] = scaler_df_generator.transform(x_train['df_generator'])
    x_train['df_hydraulic'] = scaler_df_hydraulic.transform(x_train['df_hydraulic'])
    x_train['df_gen_bear'] = scaler_df_gen_bear.transform(x_train['df_gen_bear'])
    x_train['df_transformer'] = scaler_df_transformer.transform(x_train['df_transformer'])
    x_train['df_gearbox'] = scaler_df_gearbox.transform(x_train['df_gearbox'])

    print('016 - aplicar o scale com o fit efectuado no teste')
    x_test['df_generator'] = scaler_df_generator.transform(x_test['df_generator'])
    x_test['df_hydraulic'] = scaler_df_hydraulic.transform(x_test['df_hydraulic'])
    x_test['df_gen_bear'] = scaler_df_gen_bear.transform(x_test['df_gen_bear'])
    x_test['df_transformer'] = scaler_df_transformer.transform(x_test['df_transformer'])
    x_test['df_gearbox'] = scaler_df_gearbox.transform(x_test['df_gearbox'])

    print('017 - Instanciar e treinar o modelo correspondente')
    for i in x_train:
        print(x_train[i].shape, y_train[i].shape)


    generator_model = tr(x_train=x_train['df_generator'], y_train=y_train['df_generator'], component='df_generator')
    generator_model.train()
    joblib.dump(generator_model, 'generator_model.joblib')

    hydraulic_model = tr(x_train=x_train['df_hydraulic'], y_train=y_train['df_hydraulic'], component='df_hydraulic')
    hydraulic_model.train()
    joblib.dump(hydraulic_model, 'hydraulic_model.joblib')

    gen_bear_model = tr(x_train=x_train['df_gen_bear'], y_train=y_train['df_gen_bear'], component='df_gen_bear')
    gen_bear_model.train()
    joblib.dump(gen_bear_model, 'gen_bear_model.joblib')

    transformer_model = tr(x_train=x_train['df_transformer'], y_train=y_train['df_transformer'], component='df_transformer')
    transformer_model.train()
    joblib.dump(transformer_model, 'transformer_model.joblib')

    gearbox_model = tr(x_train=x_train['df_gearbox'], y_train=y_train['df_gearbox'], component='df_gearbox')
    gearbox_model.train()
    joblib.dump(gearbox_model, 'gearbox_model.joblib')


