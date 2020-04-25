import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb')) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods =['POST'])
def predict():

    float_features = [float(x) for x in request.form.values()]
    # final_features = [np.array(float_features)]
    A=pd.DataFrame(float_features)
    A=A.iloc[:].values
    sc = StandardScaler()
    A = sc.fit_transform(A)
    B=[]
    for i in A:
        B.append(i[0])
    prediction = model.predict([B])

    return render_template('index.html', prediction_text='Probability is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)