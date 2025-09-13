from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("crop_recommendation_model.joblib")
scaler = joblib.load("scaler.joblib")

# --- Add your crop label mapping here ---
label_mapping = {
    0: "rice",
    1: "maize",
    2: "chickpea",
    3: "kidneybeans",
    4: "pigeonpeas",
    5: "mothbeans",
    6: "mungbean",
    7: "blackgram",
    8: "lentil",
    9: "pomegranate",
    10: "banana",
    11: "mango",
    12: "grapes",
    13: "watermelon",
    14: "muskmelon",
    15: "apple",
    16: "orange",
    17: "papaya",
    18: "coconut"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare data for prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_data = scaler.transform(input_data)

        # Predict numeric label
        prediction_label = model.predict(scaled_data)[0]

        # --- Convert numeric label to crop name ---
        prediction_name = label_mapping.get(prediction_label, "Unknown Crop")

        return render_template('index.html', prediction_text=f"🌾 Recommended Crop: {prediction_name}")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
