from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('iri.pkl', 'rb'))

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def home():
        
        data1 = float(request.form['a'])
        data2 = float(request.form['b'])
        data3 = float(request.form['c'])
        data4 = float(request.form['d'])
        
        arr = np.array([[data1, data2, data3, data4]])
        pred = model.predict(arr)
        
        return render_template('pred.html', data=pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
    