from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('random_forest_model.pkl')

# Define a mapping from the prediction classes to contribution amounts
class_to_amount = {
    0: 10,   
    1: 20,   
    2: 30,   
    3: 40,   
    4: 50    
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)
        result_class = prediction[0]
        result_amount = class_to_amount.get(result_class, "Unknown class")
        return jsonify({'prediction': result_amount})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


