from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('loan_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = pd.DataFrame([data])
    prediction = model.predict(features)[0]
    return jsonify({'result': 'Approved' if prediction == 1 else 'Rejected'})

if __name__ == '__main__':
    app.run(debug=True)
