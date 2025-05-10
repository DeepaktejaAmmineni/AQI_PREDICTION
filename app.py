from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the app
app = Flask(__name__)

# Load the trained XGBoost model
with open('aqi_xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Home route
@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from the form and convert to float
        features = [
            float(request.form['temp']),
            float(request.form['humidity']),
            float(request.form['pm10']),
            float(request.form['no2']),
            float(request.form['o3']),
            float(request.form['co']),
            float(request.form['so2']),
            float(request.form['wind'])
        ]

        # Reshape input for prediction
        final_features = np.array([features])
        prediction = model.predict(final_features)[0]

        return render_template(
            'index.html',
            prediction_text=f'Predicted PM2.5 Level: {prediction:.2f} µg/m³'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
