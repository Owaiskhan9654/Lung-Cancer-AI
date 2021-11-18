import joblib


def Random_forest_predict(df_predict):
    color_model='#1d0796'

    filename = 'static/Models Trained/Random_Forest_0.81_18-11-2021.sav'
    model = joblib.load(filename)
    p = model.predict_proba(df_predict)[:, 0]
    b = []
    c = model.predict(df_predict)[0]
    d = []
    if c == 0:
        c = 'Surgery Should not be Required'
        color_prediction = '#13b438'

    else:
        c = 'Yes Surgery is Required. Please Consult with your Doctor Also!!!'
        color_prediction = '#f42a2a'
    for i in p:
        if i > 0.5:
            b.append(i)
        else:
            b.append(1 - i)
    for i in range(len(b)):
        d.append(['Random Forest Model With 81% Accuracy and Trained on 18-11-2021', c, b[i],color_model,color_prediction],)

    return d[0]
def CatBoost_predict(df_predict):
    color_model='#196874'
    filename = 'static/Models Trained/CatBoost_0.81_18-11-2021.sav'
    model = joblib.load(filename)
    p = model.predict_proba(df_predict)[:, 0]
    b = []
    c = model.predict(df_predict)[0]
    d = []
    if c == 0:
        c = 'Surgery Should not be Required'
        color_prediction = '#13b438'

    else:
        c = 'Yes Surgery is Required. Please Consult with your Doctor Also!!!'
        color_prediction = '#f42a2a'
    for i in p:
        if i > 0.5:
            b.append(i)
        else:
            b.append(1 - i)
    for i in range(len(b)):
        d.append(['CatBoost Model With 81% Accuracy and Trained on 18-11-2021', c, b[i],color_model, color_prediction])

    return d[0]

def LightGBM_predict(df_predict):
    color_model = '#c4c115'
    filename = 'static/Models Trained/Light_GBM_0.81_18-11-2021.sav'
    model = joblib.load(filename)
    p = model.predict_proba(df_predict)[:, 0]
    b = []
    c = model.predict(df_predict)[0]
    d = []
    if c == 0:
        c = 'Surgery Should not be Required'
        color_prediction = '#13b438'

    else:
        c = 'Yes Surgery is Required. Please Consult with your Doctor Also!!!'
        color_prediction = '#f42a2a'
    for i in p:
        if i > 0.5:
            b.append(i)
        else:
            b.append(1 - i)
    for i in range(len(b)):
        d.append(['Light Gradient Boosting Model With 81% Accuracy and Trained on 18-11-2021', c, b[i],color_model, color_prediction])

    return d[0]


def AutoML_predict(df_predict):
    color_model='#8d02a6'
    filename = 'static/Models Trained/Auto_ML_0.81_18-11-2021.sav'
    model = joblib.load(filename)
    p = model.predict_proba(df_predict)[:, 0]
    b = []
    c = model.predict(df_predict)[0]
    d = []
    if c == 0:
        c = 'Surgery Should not be Required'
        color_prediction = '#13b438'

    else:
        c = 'Yes Surgery is Required. Please Consult with your Doctor Also!!!'
        color_prediction = '#f42a2a'
    for i in p:
        if i > 0.5:
            b.append(i)
        else:
            b.append(1 - i)
    for i in range(len(b)):
        d.append(["Auto ML Google's Model With 81% Accuracy and Trained on 18-11-2021", c, b[i],color_model, color_prediction])

    return d[0]


def Logistic_Regression_predict(df_predict):
    color_model='#CC2A1E'
    filename = 'static/Models Trained/Logistic_model_0.78_18-11-2021.sav'
    model = joblib.load(filename)
    p = model.predict_proba(df_predict)[:, 0]
    b = []
    c = model.predict(df_predict)[0]
    d = []
    if c == 0:
        c = 'Surgery Should not be Required'
        color_prediction = '#13b438'

    else:
        c = 'Yes Surgery is Required. Please Consult with your Doctor Also!!!'
        color_prediction = '#f42a2a'
    for i in p:
        if i > 0.5:
            b.append(i)
        else:
            b.append(1 - i)
    for i in range(len(b)):
        d.append(["Logistic Regression Model With 79% Accuracy and Trained on 18-11-2021", c, b[i],color_model, color_prediction])

    return d[0]

def SVC_predict(df_predict):
    color_model='#002ed4'
    filename = 'static/Models Trained/SVC_model_0.65_18-11-2021.sav'
    model = joblib.load(filename)
    p = model.predict_proba(df_predict)[:, 0]
    b = []
    c = model.predict(df_predict)[0]
    d = []
    if c == 0:
        c = 'Surgery Should not be Required'
        color_prediction = '#13b438'

    else:
        c = 'Yes Surgery is Required. Please Consult with your Doctor Also!!!'
        color_prediction = '#f42a2a'
    for i in p:
        if i > 0.5:
            b.append(i)
        else:
            b.append(1 - i)
    for i in range(len(b)):
        d.append(["Support Vector Machine Classifier With 65% Accuracy and Trained on 18-11-2021", c, b[i],color_model, color_prediction])

    return d[0]

def neural_network_predict(df_predict):
    color_model = '#e20a5a'
    filename = 'static/Models Trained/neural_network_model_0.73_18-11-2021.sav'
    model = joblib.load(filename)
    p = model.predict_proba(df_predict)[:, 0]
    b = []
    c = model.predict(df_predict)[0]
    d = []
    if c == 0:
        c = 'Surgery Should not be Required'
        color_prediction = '#13b438'

    else:
        c = 'Yes Surgery is Required. Please Consult with your Doctor Also!!!'
        color_prediction = '#f42a2a'
    for i in p:
        if i > 0.5:
            b.append(i)
        else:
            b.append(1 - i)
    for i in range(len(b)):
        d.append(["Multi Layer Perceptron (Neural Networks) With 71% Accuracy and Trained on 18-11-2021", c, b[i],color_model, color_prediction])

    return d[0]


def KNN_predict(df_predict):
    color_model = '#b62885'
    filename = 'static/Models Trained/KNN_model_0.75_18-11-2021.sav'
    model = joblib.load(filename)
    p = model.predict_proba(df_predict)[:, 0]
    b = []
    c = model.predict(df_predict)[0]
    d = []
    if c == 0:
        c = 'Surgery Should not be Required'
        color_prediction = '#13b438'

    else:
        c = 'Yes Surgery is Required. Please Consult with your Doctor Also!!!'
        color_prediction = '#f42a2a'
    for i in p:
        if i > 0.5:
            b.append(i)
        else:
            b.append(1 - i)
    for i in range(len(b)):
        d.append(["K Nearest Neighbours With 79% Accuracy and Trained on 18-11-2021", c, b[i],color_model, color_prediction])

    return d[0]

def XGB_predict(df_predict):
    color_model = '#f42cae'
    filename = 'static/Models Trained/xgboost_model_0.79_18-11-2021.sav'
    model = joblib.load(filename)
    p = model.predict_proba(df_predict)[:, 0]
    b = []
    c = model.predict(df_predict)[0]
    d = []
    if c == 0:
        c = 'Surgery Should not be Required'
        color_prediction = '#13b438'

    else:
        c = 'Yes Surgery is Required. Please Consult with your Doctor Also!!!'
        color_prediction = '#f42a2a'
    for i in p:
        if i > 0.5:
            b.append(i)
        else:
            b.append(1 - i)
    for i in range(len(b)):
        d.append(["eXtreme Gradient Boosting Model With 79% Accuracy and Trained on 18-11-2021", c, b[i],color_model, color_prediction])

    return d[0]


def extratree_predict(df_predict):
    color_model = '#840bac'
    filename = 'static/Models Trained/ExtraTree_model_0.78_18-11-2021.sav'
    model = joblib.load(filename)
    p = model.predict_proba(df_predict)[:, 0]
    b = []
    c = model.predict(df_predict)[0]
    d = []
    if c == 0:
        c = 'No Surgery Should not be Required'
        color_prediction = '#13b438'

    else:
        c = 'Yes Surgery is Required. Please Consult with your Doctor Also!!!'
        color_prediction = '#f42a2a'
    for i in p:
        if i > 0.5:
            b.append(i)
        else:
            b.append(1 - i)
    for i in range(len(b)):
        d.append(["Extra Trees Model With 77% Accuracy and Trained on 18-11-2021", c, b[i],color_model, color_prediction])

    return d[0]