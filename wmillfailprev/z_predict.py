from b_get_data import *
from c_model_related import CustomStandardScaler as Cscl
from c_model_related import Trainer as tr
from c_model_related import metrics_create_df
import joblib
from sklearn.preprocessing import StandardScaler

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
    df_dict = {'signals_df':signals_df}

    print('003 - Criar os datasets por componentes')
    df_generator, df_gen_bear, df_transformer, df_hydraulic, df_gearbox = component_df_creation(signals_df)

    print('004 - Criar o dicionário de datasets por componentes')
    comp_df_dict = {'df_generator': df_generator,'df_hydraulic': df_hydraulic,'df_gen_bear': df_gen_bear,'df_transformer': df_transformer,'df_gearbox': df_gearbox}

    print('005 - fazer cópia dos dataframes')
    comp_prep_df_dict = comp_df_dict.copy()

##### Este passo precisa de ser revisto########################################
    print('007 - Fillna by turbine')
    turbine_list = ['T11', 'T06', 'T01', 'T09', 'T07']
    for i, key in enumerate(comp_prep_df_dict):
        comp_prep_df_dict[key] = fill_na_by_turbine(comp_prep_df_dict[key],turbine_list)

    print('009 - retirar as colunas que não se relacionam com a variavel alvo')
    for i, key in enumerate(comp_prep_df_dict):
        comp_prep_df_dict[key] = comp_prep_df_dict[key].drop(columns=feat_drop_list[i])

    print('010 - agrupar pela medida de tempo seleccionada')
    for key in comp_prep_df_dict:
        comp_prep_df_dict[key] = group_por_frequency(comp_prep_df_dict[key], period='Dia')

##### Este passo precisa de ser revisto########################################
    print('013 - Adicionar medidas de alisamento')
    for key in comp_prep_df_dict:
        comp_prep_df_dict[key] = add_features(comp_prep_df_dict[key], rolling_win_size=10)

    print('013.1 - retirar a identificação das tubinas dos x_train e x_test')
    for key in comp_prep_df_dict:
        comp_prep_df_dict[key] = comp_prep_df_dict[key].drop(columns=['Turbine_ID'])

    print('014 - fazer o load do StandardScaler para cada componente à pedreiro')
    print('014.1 - df_generator')
    scaler_df_generator = joblib.load('scaler_df_generator.joblib')
    print('014.2 - df_hydraulic')
    scaler_df_hydraulic = joblib.load('scaler_df_hydraulic.joblib')
    print('014.3 - df_gen_bear')
    scaler_df_gen_bear = joblib.load('scaler_df_gen_bear.joblib')
    print('014.4 - df_transformer')
    scaler_df_transformer = joblib.load('scaler_df_transformer.joblib')
    print('014.5 - df_gearbox')
    scaler_df_gearbox = joblib.load('scaler_df_gearbox.joblib')

    print('016 - aplicar o scale com o fit efectuado no treino')
    comp_prep_df_dict['df_generator'] = scaler_df_generator.transform(comp_prep_df_dict['df_generator'])
    comp_prep_df_dict['df_hydraulic'] = scaler_df_hydraulic.transform(comp_prep_df_dict['df_hydraulic'])
    comp_prep_df_dict['df_gen_bear'] = scaler_df_gen_bear.transform(comp_prep_df_dict['df_gen_bear'])
    comp_prep_df_dict['df_transformer'] = scaler_df_transformer.transform(comp_prep_df_dict['df_transformer'])
    comp_prep_df_dict['df_gearbox'] = scaler_df_gearbox.transform(comp_prep_df_dict['df_gearbox'])

    print('017 - Instanciar e treinar o modelo correspondente')

    generator_model = joblib.load('generator_model.joblib')
    hydraulic_model = joblib.dump('hydraulic_model.joblib')
    gen_bear_model = joblib.dump('gen_bear_model.joblib')
    transformer_model = joblib.dump('transformer_model.joblib')
    gearbox_model = joblib.dump('gearbox_model.joblib')

    print('018 - Obter a previsão para a métrica de poupança')
    y_pred_generator = generator_model.predict()
    y_pred_hydraulic = hydraulic_model.predict()
    y_pred_gen_bear = gen_bear_model.predict()
    y_pred_transformer = transformer_model.predict()
    y_pred_gearbox = gearbox_model.predict()

    # print('019 - poupanças')
    # poupancas_generator, cf_numbers_pred_gen, df_resultados = metrics_create_df(df_test_comp_dict['df_generator'], y_test['df_generator'], y_pred_generator, 'df_generator', days=20)
    # print(poupancas_generator)

    # poupancas_hydraulic, cf_numbers_pred_hyd, df_resultados = metrics_create_df(df_test_comp_dict['df_hydraulic'], y_test['df_hydraulic'], y_pred_generator, 'df_hydraulic', days=20)
    # print(poupancas_hydraulic)

    # poupancas_gen_bear, cf_numbers_pred_genbear,df_resultados = metrics_create_df(df_test_comp_dict['df_gen_bear'], y_test['df_gen_bear'], y_pred_generator, 'df_gen_bear', days=20)
    # print(poupancas_gen_bear)

    # poupancas_transformer, cf_numbers_pred_transf,df_resultados = metrics_create_df(df_test_comp_dict['df_transformer'], y_test['df_transformer'], y_pred_generator, 'df_transformer', days=20)
    # print(poupancas_transformer)

    # poupancas_gearbox, cf_numbers_pred_gear, df_resultados = metrics_create_df(df_test_comp_dict['df_gearbox'], y_test['df_gearbox'], y_pred_generator, 'df_gearbox', days=20)
    # print(poupancas_gearbox, cf_numbers_pred_gear, 'gearbox')
