from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
model = joblib.load('random_forest_model.pkl')

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.info(f"Received data: {data}")

        # Ensure the data is formatted correctly
        input_data = pd.DataFrame([data])
        logging.info(f"Input data for prediction: {input_data}")

        prediction = model.predict(input_data)
        result_amount = prediction[0]
        logging.info(f"Prediction result: {result_amount}")

        return jsonify({'prediction': result_amount})
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
