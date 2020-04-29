import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb')) 

# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/predict',methods =['POST'])
def predict():

    # float_features = [float(x) for x in request.form.values()]
    final_features = dict(request.form)['data']
    # ad = [17.99	,10.38,	122.8	,1001.0,	0.1184,	0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1.095	,0.9053,	8.589	,153.4,	0.006399,	0.04904	,0.05373,	0.01587	,0.03003,	0.006193,	25.38	,17.33,	184.6,	2019.0,	0.1622,	0.6656,	0.7119,	0.2654,	0.4601]
    # A=pd.DataFrame(float_features)
    # A=A.iloc[:].values
    # sc = StandardScaler()
    # A = sc.fit_transform(A)
    # B=[]
    # for i in A:
    #     B.append(i[0])
    # prediction = model.predict([B])

    return jsonify({"response" : final_features})
    #return render_template('index.html', prediction_text='Probability is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)