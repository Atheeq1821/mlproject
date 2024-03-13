from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle

from src.pipelines.predict_pipeline import Get_Data,Predict

app=Flask(__name__)

application=app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:

        user_data=Get_Data(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        df=user_data.user_data_into_dataFrame()
        prediction=Predict()
        predicted_score=prediction.prediction_output(df)
        return render_template('home.html',results=predicted_score)
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True) 

