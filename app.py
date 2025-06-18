import os
import pandas as pd
import joblib
from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)
variable_form = ['age','gender', 'smoking', 'hx-smoking', 'hx-radiothreapy',
                 'thyroid-function', 'physical-examination','adenopathy',
                 'pathology', 'focality', 'risk', 't', 'n', 'm', 'stage',
                 'response']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result/', methods=["POST"])
def result():
    # Get features from form in index.html
    age = request.form.get('age')
    gender = request.form.get('gender')
    smoking = request.form.get('smoking')
    hx_smoking = request.form.get('hx-smoking')
    hx_radiothreapy = request.form.get('hx-radiothreapy')
    thyroid_function = request.form.get('thyroid-function')
    physical_examination = request.form.get('physical-examination')
    adenopathy = request.form.get('adenopathy')
    pathology = request.form.get('pathology')
    focality = request.form.get('focality')
    risk = request.form.get('risk')
    t = request.form.get('t')
    n = request.form.get('n')
    m = request.form.get('m')
    stage = request.form.get('stage')
    response = request.form.get('response')

    variable_names = [
        age,
        gender,
        smoking,
        hx_smoking,
        hx_radiothreapy,
        thyroid_function,
        physical_examination,
        adenopathy,
        pathology,
        focality,
        risk,
        t,
        n,
        m,
        stage,
        response,
    ]

    # Load model
    filepath = 'random_forest_model.pkl'
    load_model = joblib.load(filepath)

    # Make new dataframe for input
    df_input = pd.DataFrame(columns = variable_form)
    df_input.loc[0] = variable_names

    # Print result in result.html
    result = load_model.predict(df_input)
    for i in result:
        int_result = int(i)
        if(int_result==0):
            decision="No"
        else:
            decision="Yes"
    
    #return output
    return render_template('result.html', 
                       age=age, 
                       gender=gender, 
                       smoking=smoking, 
                       hx_smoking=hx_smoking,
                       hx_radiothreapy=hx_radiothreapy,
                       thyroid_function=thyroid_function,
                       physical_examination=physical_examination,
                       adenopathy=adenopathy,
                       pathology=pathology,
                       focality=focality,
                       risk=risk,
                       t=t,
                       n=n,
                       m=m,
                       stage=stage,
                       response=response,
                       decision=decision
                       )

if __name__ == "main":
    app.run()