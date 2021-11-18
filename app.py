from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import utils

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def landing_page():
    return render_template('index.html', show_form="false")


@app.route('/user_details', methods=['GET','POST'])
def user_page():
    year_of_diagnosis_list=[2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000]
    size_of_tumor = [i for i in range(0,1000)]

    return render_template('index.html', show_form="true",show_models='true',year_of_diagnosis_list=year_of_diagnosis_list,\
                           year_of_diagnosis_list_len=len(year_of_diagnosis_list),size_of_tumor=size_of_tumor,size_of_tumor_len=len(size_of_tumor))

@app.route('/process', methods=['POST'])
def process():
    all_data = []
    all_prediction_data=[]
    year_of_diagnosis = request.form['year_of_diagnosis']
    age_recoded = request.form['age_recoded']
    sex = request.form['sex']
    race = request.form['race']
    site_recoded = request.form['site_recoded']
    grade = request.form['grade']
    primary_site_of_cancer = request.form['pri_site_of_cancer']
    laterality = request.form['laterality']
    tumor_size = request.form['tumor_size']
    t_stage = request.form['t_stage']
    n_stage = request.form['n_stage']
    m_stage = request.form['m_stage']
    metastases_bone = request.form['metastases_bone']
    metastases_brain = request.form['metastases_brain']
    metastases_lung = request.form['metastases_lung']
    metastases_liver = request.form['metastases_liver']
    ml_algoritms = request.form.getlist('mlalgos')
    df_predict = pd.DataFrame(
        {'Agerecodewith1yearolds': [int(age_recoded)], 'RacerecodeWhiteBlackOther': [int(race)], 'Sex': [int(sex)],\
         'Yearofdiagnosis': [int(year_of_diagnosis)], 'SiterecodeICDO3WHO2008': [int(site_recoded)],\
         'PrimarySite': [int(primary_site_of_cancer)],'Gradethru2017': [int(grade)], 'Laterality': [int(laterality)],\
         'DerivedAJCCT6thed20042015': [int(t_stage)],'DerivedAJCCN6thed20042015': [int(n_stage)], \
         'DerivedAJCCM6thed20042015': [int(m_stage)],'SEERCombinedMetsatDXbone2010': [int(metastases_bone)],
         'SEERCombinedMetsatDXbrain2010': [int(metastases_brain)],'SEERCombinedMetsatDXliver2010': [int(metastases_liver)],\
         'SEERCombinedMetsatDXlung2010': [int(metastases_lung)],'CStumorsize20042015': [int(tumor_size)]})
    Prediction_list=['No Model Chosen']
    for i in ml_algoritms:

        if i == 'random_forest':
            Prediction_list = utils.Random_forest_predict(df_predict)

        elif i == 'catboost':
            Prediction_list = utils.CatBoost_predict(df_predict)

        elif i == 'lightgbm':
            Prediction_list = utils.LightGBM_predict(df_predict)

        elif i == 'automl':
            Prediction_list = utils.AutoML_predict(df_predict)

        elif i == 'logisticregression':
            Prediction_list = utils.Logistic_Regression_predict(df_predict)

        elif i == 'svm':
            Prediction_list = utils.SVC_predict(df_predict)

        elif i == 'neural_network':
            Prediction_list = utils.neural_network_predict(df_predict)

        elif i == 'knn':
            Prediction_list = utils.KNN_predict(df_predict)

        elif i == 'xgboost':
            Prediction_list = utils.XGB_predict(df_predict)

        elif i == 'extratree':
            Prediction_list= utils.extratree_predict(df_predict)

        all_prediction_data.append(Prediction_list)


    print(all_prediction_data)
    return render_template('predict.html',show_error='false',all_prediction_data=all_prediction_data)

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1")
