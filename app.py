from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model and label encoder
model = joblib.load('model/mental_health_model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

@app.route('/')
def index():
    return render_template('survey.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Collect form data
    data = {
        'interest': [int(request.form['interest'])],
        'depressed': [int(request.form['depressed'])],
        'sleep': [int(request.form['sleep'])],
        'fatigue': [int(request.form['fatigue'])],
        'appetite': [int(request.form['appetite'])],
        'worthlessness': [int(request.form['worthlessness'])],
        'concentration': [int(request.form['concentration'])],
        'restlessness': [int(request.form['restlessness'])],
        'suicidal': [int(request.form['suicidal'])]
    }
    
    df = pd.DataFrame(data)
    
    # Predict disorder
    prediction = model.predict(df)
    decoded_prediction = label_encoder.inverse_transform(prediction)
    return f'Predicted Disorder: {decoded_prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)
