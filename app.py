# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'Project3.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        ty = int(request.form['type'])
        at = float(request.form['airtemp'])
        pt = float(request.form['proctemp'])
        rp = float(request.form['rotspeed'])
        tw = float(request.form['toolwear'])
        twf = int(request.form['twf'])
        hdf = int(request.form['hdf'])
        pwf = int(request.form['pwf'])
        osf = int(request.form['osf'])
        rnf = int(request.form['rnf'])
        
        data = np.array([[ty,at,pt,rp,tw,twf,hdf,pwf,osf,rnf]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)