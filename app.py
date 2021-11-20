from flask import Flask, render_template, request, send_from_directory,redirect,url_for,session
from datetime import timedelta
import pandas as pd
import utils
import functools
import os
app = Flask(__name__)
app.secret_key = "CanaryGlobal@2021"
app.config['PERMANENT_SESSION_LIFETIME'] =  timedelta(minutes=10)
def login_required(func):
    @functools.wraps(func)
    def secure_function():
        if "username" not in session:
            error_login = 'Please Login to acess this Page'
            return redirect(url_for("login_page",error_login=error_login))
        return func()
    return secure_function
@app.route('/', methods=['GET', 'POST'])
def login_page():
    error_login = None
    if request.method == 'POST':
        users = {'owais.ahmed@canarydetect.com': 'Canary@2021', 'r.varsha@canarydetect.com': 'Canary@2021',\
             'prince.nadar@canarydetect.com':'Canary@2021','hashmi.farogh@canarydetect.com':'Canary@2021',\
             'geetesh.mishra@canarydetect.com':'Canary@2021','purav.badani@canarydetect.com':'Canary@2021',\
             'milind.tamore@canarydetect.com':'Canary@2021','zohaib.ahmed@canarydetect.com':'Canary@2021',\
                'gopal.palla@canarydetect.com':'Canary@2021','raj@canarydetect.com':'Canary@2021',\
                 'harshavardhan.karkar@canarydetect.com':'Canary@2021','avinash.joshi@canarydetect.com':'Canary@2021',\
                 'osho.sachdeva@canarydetect.com':'Canary@2021','prashant.ghodwade@canarydetect.com':'Canary@2021',\
                 'mj@canarydetect.com':'Canary@2021','ashwin.chalke@canarydetect.com':'Canary@2021',\
                 'shirish.deshpande@canarydetect.com':'Canary@2021'}
        if request.form['username'] in users and users[request.form['username']] == request.form['password']:
            session["username"] = request.form['username']
            return redirect(url_for('landing_page_after_login'))

        elif request.form['username'] not in users:
            return render_template('login.html', error_login='User is Not in DataBase..Please Contact Admin')

        else:
            return render_template('login.html', error_login='Invalid password provided')

    return render_template('login.html', error_login=None)
@app.route('/after_login', methods=['GET', 'POST'])
@login_required
def landing_page_after_login():
    return render_template('index.html', show_form="false")





@app.route('/loading', methods=['GET', 'POST'])
@login_required
def loading_page():
    try:
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
        df_data = pd.DataFrame(
            {'Agerecodewith1yearolds': [int(age_recoded)], 'RacerecodeWhiteBlackOther': [int(race)], 'Sex': [int(sex)],
             'Yearofdiagnosis': [int(year_of_diagnosis)], 'SiterecodeICDO3WHO2008': [int(site_recoded)],
             'PrimarySite': [int(primary_site_of_cancer)], 'Gradethru2017': [int(grade)], 'Laterality': [int(laterality)],
             'DerivedAJCCT6thed20042015': [int(t_stage)], 'DerivedAJCCN6thed20042015': [int(n_stage)],
             'DerivedAJCCM6thed20042015': [int(m_stage)], 'SEERCombinedMetsatDXbone2010': [int(metastases_bone)],
             'SEERCombinedMetsatDXbrain2010': [int(metastases_brain)],
             'SEERCombinedMetsatDXliver2010': [int(metastases_liver)],
             'SEERCombinedMetsatDXlung2010': [int(metastases_lung)], 'CStumorsize20042015': [int(tumor_size)]})

        df_ml_algorithms=pd.DataFrame({'ml_algoritms': ml_algoritms})


        df_data.to_csv('static/tempdf.csv',index=False)
        df_ml_algorithms.to_csv('static/tempml.csv',index=False)
        return render_template('loading.html',)
    except:
        all_prediction_data = []
        return render_template('predict.html', show_error='true', all_prediction_data=all_prediction_data)

@app.route('/user_details', methods=['GET','POST'])
@login_required
def user_page():
    year_of_diagnosis_list=[2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000]
    size_of_tumor = [i for i in range(0,1000)]

    return render_template('index.html', show_form="true",show_models='true',year_of_diagnosis_list=year_of_diagnosis_list,\
                           year_of_diagnosis_list_len=len(year_of_diagnosis_list),size_of_tumor=size_of_tumor,size_of_tumor_len=len(size_of_tumor))

@app.route('/process', methods=['GET','POST'])
@login_required
def process():
    all_data = []
    all_prediction_data=[]

    try:
        df_data_predict = pd.read_csv('static/tempdf.csv')
        df_algorithms = pd.read_csv('static/tempml.csv')
        os.remove('static/tempml.csv')
        os.remove('static/tempdf.csv')
    except:
        return render_template('predict.html', show_error='true', all_prediction_data=all_prediction_data)
    try:
        ml_algoritms=list(df_algorithms['ml_algoritms'])

        Prediction_list=['No Model Chosen']
        for i in ml_algoritms:

            if i == 'random_forest':
                Prediction_list = utils.Random_forest_predict(df_data_predict)

            elif i == 'catboost':
                Prediction_list = utils.CatBoost_predict(df_data_predict)

            elif i == 'lightgbm':
                Prediction_list = utils.LightGBM_predict(df_data_predict)

            elif i == 'automl':
                Prediction_list = utils.AutoML_predict(df_data_predict)

            elif i == 'logisticregression':
                Prediction_list = utils.Logistic_Regression_predict(df_data_predict)

            elif i == 'svm':
                Prediction_list = utils.SVC_predict(df_data_predict)

            elif i == 'neural_network':
                Prediction_list = utils.neural_network_predict(df_data_predict)

            elif i == 'knn':
                Prediction_list = utils.KNN_predict(df_data_predict)

            elif i == 'xgboost':
                Prediction_list = utils.XGB_predict(df_data_predict)

            elif i == 'extratree':
                Prediction_list= utils.extratree_predict(df_data_predict)

            all_prediction_data.append(Prediction_list)


        return render_template('predict.html',show_error='false',all_prediction_data=all_prediction_data)
    except:
        all_prediction_data=[]
        return render_template('predict.html', show_error='true', all_prediction_data=all_prediction_data)

@app.route('/logout', methods=['GET','POST'])
def logout():
    session.clear()
    return redirect(url_for('login_page'))

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1")
