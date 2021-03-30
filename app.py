
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            Type=float(request.form['Type'])
            Air_temperature_[K] = float(request.form['Air_temperature_[K]'])
            Process_temperature_[K] = float(request.form['Process_temperature_[K]'])
            Rotational_speed_[rpm] = float(request.form['Rotational_speed_[rpm]'])
            Tool_wear_[min] = float(request.form['Tool_wear_[min]'])
            TWF = float(request.form['TWF'])
            HDF = float(request.form['HDF'])
            PWF = float(request.form['PWF'])
            OSF = float(request.form['OSF'])
            RNF = float(request.form['RNF'])
            filename = 'Project3(SL).pkl'
            loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage
            # predictions using the loaded model file
            prediction=loaded_model.predict([[Type,Air_temperature_[K],Process_temperature_[K],Rotational_speed_[rpm],Tool_wear_[min],TWF,HDF,PWF,OSF,RNF]])
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=round(100*prediction[0]))
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app