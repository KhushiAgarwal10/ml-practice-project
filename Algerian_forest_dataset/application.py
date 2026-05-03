from flask import Flask,render_template,request,jsonify,url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

# import ridge regression and standardscaler
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scale.pkl','rb'))



@app.route('/')
def index():
  return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_data():
  if request.method=='POST':
    FWI=float(request.form.get('FWI'))
    RH=float(request.form.get('RH'))
    Ws=float(request.form.get('Ws'))
    Rain=float(request.form.get('Rain'))
    FFMC=float(request.form.get('FFMC'))
    DMC=float(request.form.get('DMC'))
    ISI=float(request.form.get('ISI'))
    Classes=float(request.form.get('Classes'))
    Region=float(request.form.get('Region'))

    new_data_scale=standard_scaler.transform([[FWI,RH,Ws,Rain,FFMC,DMC,ISI,Classes]])
    result=ridge_model.predict(new_data_scale)
    return render_template('home.html',results=result[0])
  else:
    return render_template('home.html')



if __name__=="__main__":
  app.run(debug=True)