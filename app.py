import numpy as np
from flask import Flask,request,jsonify,render_template,url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('model_titanic.pkl','rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	features=[str(x) for x in request.form.values()]
	print(features)
	final = [np.array(features)]
	prediction = model.predict(final)

	if prediction==0:
		tt = "Sorry,You Won't be able to Purchase!!"
	else:
		tt= "Yes Let's Have some Purchasing"

	return render_template('index.html',prediction_text='{}'.format(tt))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
