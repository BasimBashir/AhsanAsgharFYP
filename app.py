import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Creating Flask app
app = Flask(__name__)

# Loading the pickle model
model = pickle.load(open('model.pkl', 'rb'))


# Home page landing
@app.route('/')
def home():
    return render_template('index.html')


# predicting the results from html form
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html',
                           prediction_text='Daraz product score (in range of 1-10) is:  {}'.format(output))


# prediction using api call
@app.route('/calculate', methods=['POST'])
def calculate():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    prediction = model.predict(query_df)
    return jsonify({'prediction': list(prediction)})


# main function
if __name__ == "__main__":
    app.run(debug=True)
