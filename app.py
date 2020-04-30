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

    final_features = [dict(request.form)['radiusMean'] , dict(request.form)['textureMean'], dict(request.form)['perimeterMean'],dict(request.form)['areaMean'] , dict(request.form)['smoothnessMean'], dict(request.form)['compactnessMean'],dict(request.form)['concavityMean'] , dict(request.form)['concavePointsMean'], dict(request.form)['symmetryMean']
                        ,dict(request.form)['fractalDimensionMean'] , dict(request.form)['radiusSe'], dict(request.form),['textureSe'],dict(request.form)['perimeterSe'] , dict(request.form)['areaSe'], dict(request.form)['smoothnessSe'],
                        dict(request.form)['compactnessSe'] , dict(request.form)['concavitySe'], dict(request.form)['concavePointsSe'],dict(request.form)['symmetrySe'] , dict(request.form)['fractalDimensionSe'], dict(request.form)['radiusWorst'],dict(request.form)['textureWorst'] , dict(request.form)['perimeterWorst'], dict(request.form)['areaWorst'],
                        dict(request.form)['smoothnessWorst'] , dict(request.form)['compactnessWorst'], dict(request.form)['concavityWorst'],dict(request.form)['concavePointsWorst'] , dict(request.form)['symmetryWorst']]
    # ad = [17.99	,10.38,	122.8	,1001.0,	0.1184,	0.2776,	0.3001,	0.1471,	0.2419,	0.07871,	1.095	,0.9053,	8.589	,153.4,	0.006399,	0.04904	,0.05373,	0.01587	,0.03003,	0.006193,	25.38	,17.33,	184.6,	2019.0,	0.1622,	0.6656,	0.7119,	0.2654,	0.4601]
    float_features = [float(x) for x in final_features]
    A=pd.DataFrame(float_features)
    A=A.iloc[:].values
    sc = StandardScaler()
    A = sc.fit_transform(A)
    B=[]
    for i in A:
        B.append(i[0])
    prediction = model.predict([B])

    return jsonify({"response" : float_features[1]})
    #return render_template('index.html', prediction_text='Probability is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)