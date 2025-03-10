from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained ML model
with open('irrigation_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict_irrigation():
    try:
        data = request.json
        crop_type = data['crop_type']
        soil_moisture = data['soil_moisture']
        temperature = data['temperature']
        humidity = data['humidity']
        pH = data['pH']
        
        # Convert categorical crop_type into numerical if needed (dummy encoding)
        crop_mapping = {'wheat': 0, 'rice': 1, 'corn': 2}  # Adjust as per dataset
        crop_value = crop_mapping.get(crop_type, -1)
        
        if crop_value == -1:
            return jsonify({'error': 'Invalid crop type'}), 400
        
        # Create input array for prediction
        input_features = np.array([[crop_value, soil_moisture, temperature, humidity, pH]])
        
        # Predict irrigation schedule
        prediction = model.predict(input_features)
        
        return jsonify({'next_irrigation_days': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
