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
    signals_path = os.path.join(csv_path, 'wind-farm-1-signals-testing.csv')
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

    print('007 - Fillna by turbine')
    turbine_list = ['T11', 'T06', 'T01', 'T09', 'T07']
    for i, key in enumerate(comp_prep_df_dict):
        comp_prep_df_dict[key] = fill_na_by_turb_predict(comp_prep_df_dict[key],turbine_list)

    print('009 - retirar as colunas que não se relacionam com a variavel alvo')
    for i, key in enumerate(comp_prep_df_dict):
        comp_prep_df_dict[key] = comp_prep_df_dict[key].drop(columns=feat_drop_list[i])

    print('010 - agrupar pela medida de tempo seleccionada')
    for key in comp_prep_df_dict:
        comp_prep_df_dict[key] = group_por_frequency(comp_prep_df_dict[key], period='Dia')

    print('013 - Adicionar medidas de alisamento')
    for key in comp_prep_df_dict:
        comp_prep_df_dict[key] = add_feat_predict(comp_prep_df_dict[key], rolling_win_size=10)

    print('013.1 - retirar a identificação das tubinas dos x_train e x_test')
    for key in comp_prep_df_dict:
        comp_prep_df_dict[key] = comp_prep_df_dict[key].drop(columns=['Turbine_ID'])

    print('014 - Colocar as colunas pela ordem correcta')
    cols_order_generator = [
    'Amb_Temp_Avg', 'Amb_WindDir_Abs_Avg', 'Amb_WindDir_Relative_Avg',
    'Amb_WindSpeed_Avg', 'Amb_WindSpeed_Max', 'Amb_WindSpeed_Min',
    'Amb_WindSpeed_Std', 'Blds_PitchAngle_Avg', 'Blds_PitchAngle_Std',
    'Cont_Top_Temp_Avg', 'Cont_VCP_ChokcoilTemp_Avg', 'Cont_VCP_WtrTemp_Avg',
    'Gen_Bear2_Temp_Avg', 'Gen_Bear_Temp_Avg', 'Gen_Phase2_Temp_Avg',
    'Gen_RPM_Avg', 'Gen_RPM_Std', 'Gen_SlipRing_Temp_Avg',
    'Grd_Busbar_Temp_Avg', 'Grd_RtrInvPhase2_Temp_Avg', 'Hyd_Oil_Temp_Avg',
    'Nac_Direction_Avg', 'Nac_Temp_Avg', 'Prod_LatestAvg_ActPwrGen0',
    'Prod_LatestAvg_ActPwrGen1', 'Prod_LatestAvg_ReactPwrGen0',
    'Amb_Temp_Avg_av', 'Amb_WindDir_Abs_Avg_av', 'Amb_WindDir_Relative_Avg_av',
    'Amb_WindSpeed_Avg_av', 'Amb_WindSpeed_Max_av', 'Amb_WindSpeed_Min_av',
    'Amb_WindSpeed_Std_av', 'Blds_PitchAngle_Avg_av', 'Blds_PitchAngle_Std_av',
    'Cont_Top_Temp_Avg_av', 'Cont_VCP_ChokcoilTemp_Avg_av',
    'Cont_VCP_WtrTemp_Avg_av', 'Gen_Bear2_Temp_Avg_av', 'Gen_Bear_Temp_Avg_av',
    'Gen_Phase2_Temp_Avg_av', 'Gen_RPM_Avg_av', 'Gen_RPM_Std_av',
    'Gen_SlipRing_Temp_Avg_av', 'Grd_Busbar_Temp_Avg_av',
    'Grd_RtrInvPhase2_Temp_Avg_av', 'Hyd_Oil_Temp_Avg_av',
    'Nac_Direction_Avg_av', 'Nac_Temp_Avg_av', 'Prod_LatestAvg_ActPwrGen0_av',
    'Prod_LatestAvg_ActPwrGen1_av', 'Prod_LatestAvg_ReactPwrGen0_av',
    'Amb_Temp_Avg_sd', 'Amb_WindDir_Abs_Avg_sd', 'Amb_WindDir_Relative_Avg_sd',
    'Amb_WindSpeed_Avg_sd', 'Amb_WindSpeed_Max_sd', 'Amb_WindSpeed_Min_sd',
    'Amb_WindSpeed_Std_sd', 'Blds_PitchAngle_Avg_sd', 'Blds_PitchAngle_Std_sd',
    'Cont_Top_Temp_Avg_sd', 'Cont_VCP_ChokcoilTemp_Avg_sd',
    'Cont_VCP_WtrTemp_Avg_sd', 'Gen_Bear2_Temp_Avg_sd', 'Gen_Bear_Temp_Avg_sd',
    'Gen_Phase2_Temp_Avg_sd', 'Gen_RPM_Avg_sd', 'Gen_RPM_Std_sd',
    'Gen_SlipRing_Temp_Avg_sd', 'Grd_Busbar_Temp_Avg_sd',
    'Grd_RtrInvPhase2_Temp_Avg_sd', 'Hyd_Oil_Temp_Avg_sd',
    'Nac_Direction_Avg_sd', 'Nac_Temp_Avg_sd', 'Prod_LatestAvg_ActPwrGen0_sd',
    'Prod_LatestAvg_ActPwrGen1_sd', 'Prod_LatestAvg_ReactPwrGen0_sd']

    comp_prep_df_dict['df_generator'] = comp_prep_df_dict['df_generator'][cols_order_generator]

    cols_order_hydraulic = [
    'Amb_Temp_Avg', 'Amb_WindDir_Abs_Avg', 'Amb_WindDir_Relative_Avg',
    'Amb_WindSpeed_Avg', 'Amb_WindSpeed_Max', 'Amb_WindSpeed_Min',
    'Amb_WindSpeed_Std', 'Blds_PitchAngle_Avg', 'Blds_PitchAngle_Std',
    'Cont_Top_Temp_Avg', 'Cont_VCP_ChokcoilTemp_Avg', 'Cont_VCP_WtrTemp_Avg',
    'Grd_Prod_VoltPhse1_Avg', 'Grd_Prod_VoltPhse2_Avg',
    'Grd_Prod_VoltPhse3_Avg', 'Grd_RtrInvPhase2_Temp_Avg', 'Hyd_Oil_Temp_Avg',
    'Nac_Direction_Avg', 'Nac_Temp_Avg', 'Rtr_RPM_Avg', 'Rtr_RPM_Std',
    'Amb_Temp_Avg_av', 'Amb_WindDir_Abs_Avg_av', 'Amb_WindDir_Relative_Avg_av',
    'Amb_WindSpeed_Avg_av', 'Amb_WindSpeed_Max_av', 'Amb_WindSpeed_Min_av',
    'Amb_WindSpeed_Std_av', 'Blds_PitchAngle_Avg_av', 'Blds_PitchAngle_Std_av',
    'Cont_Top_Temp_Avg_av', 'Cont_VCP_ChokcoilTemp_Avg_av',
    'Cont_VCP_WtrTemp_Avg_av', 'Grd_Prod_VoltPhse1_Avg_av',
    'Grd_Prod_VoltPhse2_Avg_av', 'Grd_Prod_VoltPhse3_Avg_av',
    'Grd_RtrInvPhase2_Temp_Avg_av', 'Hyd_Oil_Temp_Avg_av',
    'Nac_Direction_Avg_av', 'Nac_Temp_Avg_av', 'Rtr_RPM_Avg_av',
    'Rtr_RPM_Std_av', 'Amb_Temp_Avg_sd', 'Amb_WindDir_Abs_Avg_sd',
    'Amb_WindDir_Relative_Avg_sd', 'Amb_WindSpeed_Avg_sd',
    'Amb_WindSpeed_Max_sd', 'Amb_WindSpeed_Min_sd', 'Amb_WindSpeed_Std_sd',
    'Blds_PitchAngle_Avg_sd', 'Blds_PitchAngle_Std_sd', 'Cont_Top_Temp_Avg_sd',
    'Cont_VCP_ChokcoilTemp_Avg_sd', 'Cont_VCP_WtrTemp_Avg_sd',
    'Grd_Prod_VoltPhse1_Avg_sd', 'Grd_Prod_VoltPhse2_Avg_sd',
    'Grd_Prod_VoltPhse3_Avg_sd', 'Grd_RtrInvPhase2_Temp_Avg_sd',
    'Hyd_Oil_Temp_Avg_sd', 'Nac_Direction_Avg_sd', 'Nac_Temp_Avg_sd',
    'Rtr_RPM_Avg_sd', 'Rtr_RPM_Std_sd']

    comp_prep_df_dict['df_hydraulic'] = comp_prep_df_dict['df_hydraulic'][cols_order_hydraulic]

    cols_order_gen_bear = [
    'Amb_Temp_Avg', 'Amb_WindDir_Abs_Avg', 'Amb_WindDir_Relative_Avg',
    'Amb_WindSpeed_Avg', 'Amb_WindSpeed_Max', 'Amb_WindSpeed_Min',
    'Amb_WindSpeed_Std', 'Blds_PitchAngle_Avg', 'Blds_PitchAngle_Std',
    'Cont_Top_Temp_Avg', 'Cont_VCP_ChokcoilTemp_Avg', 'Cont_VCP_WtrTemp_Avg',
    'Gen_Bear2_Temp_Avg', 'Gen_Bear_Temp_Avg', 'Gen_Phase2_Temp_Avg',
    'Gen_RPM_Avg', 'Gen_RPM_Std', 'Gen_SlipRing_Temp_Avg',
    'Grd_Busbar_Temp_Avg', 'Grd_RtrInvPhase2_Temp_Avg', 'Hyd_Oil_Temp_Avg',
    'Nac_Direction_Avg', 'Nac_Temp_Avg', 'Prod_LatestAvg_ActPwrGen0',
    'Prod_LatestAvg_ActPwrGen1', 'Prod_LatestAvg_ReactPwrGen0',
    'Amb_Temp_Avg_av', 'Amb_WindDir_Abs_Avg_av', 'Amb_WindDir_Relative_Avg_av',
    'Amb_WindSpeed_Avg_av', 'Amb_WindSpeed_Max_av', 'Amb_WindSpeed_Min_av',
    'Amb_WindSpeed_Std_av', 'Blds_PitchAngle_Avg_av', 'Blds_PitchAngle_Std_av',
    'Cont_Top_Temp_Avg_av', 'Cont_VCP_ChokcoilTemp_Avg_av',
    'Cont_VCP_WtrTemp_Avg_av', 'Gen_Bear2_Temp_Avg_av', 'Gen_Bear_Temp_Avg_av',
    'Gen_Phase2_Temp_Avg_av', 'Gen_RPM_Avg_av', 'Gen_RPM_Std_av',
    'Gen_SlipRing_Temp_Avg_av', 'Grd_Busbar_Temp_Avg_av',
    'Grd_RtrInvPhase2_Temp_Avg_av', 'Hyd_Oil_Temp_Avg_av',
    'Nac_Direction_Avg_av', 'Nac_Temp_Avg_av', 'Prod_LatestAvg_ActPwrGen0_av',
    'Prod_LatestAvg_ActPwrGen1_av', 'Prod_LatestAvg_ReactPwrGen0_av',
    'Amb_Temp_Avg_sd', 'Amb_WindDir_Abs_Avg_sd', 'Amb_WindDir_Relative_Avg_sd',
    'Amb_WindSpeed_Avg_sd', 'Amb_WindSpeed_Max_sd', 'Amb_WindSpeed_Min_sd',
    'Amb_WindSpeed_Std_sd', 'Blds_PitchAngle_Avg_sd', 'Blds_PitchAngle_Std_sd',
    'Cont_Top_Temp_Avg_sd', 'Cont_VCP_ChokcoilTemp_Avg_sd',
    'Cont_VCP_WtrTemp_Avg_sd', 'Gen_Bear2_Temp_Avg_sd', 'Gen_Bear_Temp_Avg_sd',
    'Gen_Phase2_Temp_Avg_sd', 'Gen_RPM_Avg_sd', 'Gen_RPM_Std_sd',
    'Gen_SlipRing_Temp_Avg_sd', 'Grd_Busbar_Temp_Avg_sd',
    'Grd_RtrInvPhase2_Temp_Avg_sd', 'Hyd_Oil_Temp_Avg_sd',
    'Nac_Direction_Avg_sd', 'Nac_Temp_Avg_sd', 'Prod_LatestAvg_ActPwrGen0_sd',
    'Prod_LatestAvg_ActPwrGen1_sd', 'Prod_LatestAvg_ReactPwrGen0_sd']

    comp_prep_df_dict['df_gen_bear'] = comp_prep_df_dict['df_gen_bear'][cols_order_gen_bear]

    cols_order_transformer = [
    'Amb_Temp_Avg', 'Amb_WindDir_Abs_Avg', 'Amb_WindDir_Relative_Avg',
    'Amb_WindSpeed_Avg', 'Amb_WindSpeed_Max', 'Amb_WindSpeed_Min',
    'Amb_WindSpeed_Std', 'Blds_PitchAngle_Avg', 'Blds_PitchAngle_Std',
    'Cont_Hub_Temp_Avg', 'Cont_Top_Temp_Avg', 'Cont_VCP_ChokcoilTemp_Avg',
    'Cont_VCP_WtrTemp_Avg', 'Grd_Busbar_Temp_Avg', 'Grd_Prod_VoltPhse1_Avg',
    'Grd_Prod_VoltPhse2_Avg', 'Grd_Prod_VoltPhse3_Avg',
    'Grd_RtrInvPhase2_Temp_Avg', 'HVTrafo_Phase2_Temp_Avg',
    'Nac_Direction_Avg', 'Nac_Temp_Avg', 'Rtr_RPM_Avg', 'Rtr_RPM_Std',
    'Amb_Temp_Avg_av', 'Amb_WindDir_Abs_Avg_av', 'Amb_WindDir_Relative_Avg_av',
    'Amb_WindSpeed_Avg_av', 'Amb_WindSpeed_Max_av', 'Amb_WindSpeed_Min_av',
    'Amb_WindSpeed_Std_av', 'Blds_PitchAngle_Avg_av', 'Blds_PitchAngle_Std_av',
    'Cont_Hub_Temp_Avg_av', 'Cont_Top_Temp_Avg_av',
    'Cont_VCP_ChokcoilTemp_Avg_av', 'Cont_VCP_WtrTemp_Avg_av',
    'Grd_Busbar_Temp_Avg_av', 'Grd_Prod_VoltPhse1_Avg_av',
    'Grd_Prod_VoltPhse2_Avg_av', 'Grd_Prod_VoltPhse3_Avg_av',
    'Grd_RtrInvPhase2_Temp_Avg_av', 'HVTrafo_Phase2_Temp_Avg_av',
    'Nac_Direction_Avg_av', 'Nac_Temp_Avg_av', 'Rtr_RPM_Avg_av',
    'Rtr_RPM_Std_av', 'Amb_Temp_Avg_sd', 'Amb_WindDir_Abs_Avg_sd',
    'Amb_WindDir_Relative_Avg_sd', 'Amb_WindSpeed_Avg_sd',
    'Amb_WindSpeed_Max_sd', 'Amb_WindSpeed_Min_sd', 'Amb_WindSpeed_Std_sd',
    'Blds_PitchAngle_Avg_sd', 'Blds_PitchAngle_Std_sd', 'Cont_Hub_Temp_Avg_sd',
    'Cont_Top_Temp_Avg_sd', 'Cont_VCP_ChokcoilTemp_Avg_sd',
    'Cont_VCP_WtrTemp_Avg_sd', 'Grd_Busbar_Temp_Avg_sd',
    'Grd_Prod_VoltPhse1_Avg_sd', 'Grd_Prod_VoltPhse2_Avg_sd',
    'Grd_Prod_VoltPhse3_Avg_sd', 'Grd_RtrInvPhase2_Temp_Avg_sd',
    'HVTrafo_Phase2_Temp_Avg_sd', 'Nac_Direction_Avg_sd', 'Nac_Temp_Avg_sd',
    'Rtr_RPM_Avg_sd', 'Rtr_RPM_Std_sd']

    comp_prep_df_dict['df_transformer'] = comp_prep_df_dict['df_transformer'][cols_order_transformer]

    cols_order_gearbox = [
    'Amb_Temp_Avg', 'Amb_WindDir_Abs_Avg', 'Amb_WindDir_Relative_Avg',
    'Amb_WindSpeed_Avg', 'Amb_WindSpeed_Max', 'Amb_WindSpeed_Min',
    'Amb_WindSpeed_Std', 'Blds_PitchAngle_Avg', 'Blds_PitchAngle_Std',
    'Cont_Hub_Temp_Avg', 'Cont_Top_Temp_Avg', 'Cont_VCP_ChokcoilTemp_Avg',
    'Cont_VCP_WtrTemp_Avg', 'Gear_Bear_Temp_Avg', 'Gear_Oil_Temp_Avg',
    'Grd_RtrInvPhase2_Temp_Avg', 'Hyd_Oil_Temp_Avg', 'Nac_Direction_Avg',
    'Nac_Temp_Avg', 'Rtr_RPM_Avg', 'Rtr_RPM_Std', 'Amb_Temp_Avg_av',
    'Amb_WindDir_Abs_Avg_av', 'Amb_WindDir_Relative_Avg_av',
    'Amb_WindSpeed_Avg_av', 'Amb_WindSpeed_Max_av', 'Amb_WindSpeed_Min_av',
    'Amb_WindSpeed_Std_av', 'Blds_PitchAngle_Avg_av', 'Blds_PitchAngle_Std_av',
    'Cont_Hub_Temp_Avg_av', 'Cont_Top_Temp_Avg_av',
    'Cont_VCP_ChokcoilTemp_Avg_av', 'Cont_VCP_WtrTemp_Avg_av',
    'Gear_Bear_Temp_Avg_av', 'Gear_Oil_Temp_Avg_av',
    'Grd_RtrInvPhase2_Temp_Avg_av', 'Hyd_Oil_Temp_Avg_av',
    'Nac_Direction_Avg_av', 'Nac_Temp_Avg_av', 'Rtr_RPM_Avg_av',
    'Rtr_RPM_Std_av', 'Amb_Temp_Avg_sd', 'Amb_WindDir_Abs_Avg_sd',
    'Amb_WindDir_Relative_Avg_sd', 'Amb_WindSpeed_Avg_sd',
    'Amb_WindSpeed_Max_sd', 'Amb_WindSpeed_Min_sd', 'Amb_WindSpeed_Std_sd',
    'Blds_PitchAngle_Avg_sd', 'Blds_PitchAngle_Std_sd', 'Cont_Hub_Temp_Avg_sd',
    'Cont_Top_Temp_Avg_sd', 'Cont_VCP_ChokcoilTemp_Avg_sd',
    'Cont_VCP_WtrTemp_Avg_sd', 'Gear_Bear_Temp_Avg_sd', 'Gear_Oil_Temp_Avg_sd',
    'Grd_RtrInvPhase2_Temp_Avg_sd', 'Hyd_Oil_Temp_Avg_sd',
    'Nac_Direction_Avg_sd', 'Nac_Temp_Avg_sd', 'Rtr_RPM_Avg_sd',
    'Rtr_RPM_Std_sd']

    comp_prep_df_dict['df_gearbox'] = comp_prep_df_dict['df_gearbox'][cols_order_gearbox]

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
    hydraulic_model = joblib.load('hydraulic_model.joblib')
    gen_bear_model = joblib.load('gen_bear_model.joblib')
    transformer_model = joblib.load('transformer_model.joblib')
    gearbox_model = joblib.load('gearbox_model.joblib')

    print('018 - Obter a previsão para a métrica de poupança')
    y_pred_generator = generator_model.predict(comp_prep_df_dict['df_generator'])
    y_pred_hydraulic = hydraulic_model.predict(comp_prep_df_dict['df_hydraulic'])
    y_pred_gen_bear = gen_bear_model.predict(comp_prep_df_dict['df_gen_bear'])
    y_pred_transformer = transformer_model.predict(comp_prep_df_dict['df_transformer'])
    y_pred_gearbox = gearbox_model.predict(comp_prep_df_dict['df_gearbox'])
    print(y_pred_generator)
