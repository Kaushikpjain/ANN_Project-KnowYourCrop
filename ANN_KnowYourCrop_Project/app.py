# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import logging
import threading
import webbrowser
import time
import tensorflow as tf
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)


class CropRecommendationSystem:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        self.is_trained = False
        # Corrected paths - match exactly what your training code saves
        self.model_save_path = 'saved_models/crop_recommendation_model.h5'
        self.scaler_save_path = 'saved_models/scaler.pkl'
        self.encoder_save_path = 'saved_models/label_encoder.pkl'
        self.feature_names_path = 'saved_models/feature_names.pkl'

    def load_trained_model(self):
        """Load the pre-trained model and preprocessing objects from the new training script"""
        try:
            print("🔍 Checking for trained model files...")

            # Check if all required files exist
            required_files = [
                self.model_save_path,
                self.scaler_save_path,
                self.encoder_save_path,
                self.feature_names_path
            ]

            missing_files = []
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                    print(f"❌ Missing: {file_path}")

            if missing_files:
                print(f"❌ Missing {len(missing_files)} required files. Please run the training script first.")
                print("💡 Run: python crop_recommendation_ann.py")
                return False

            print("📂 Loading pre-trained model...")

            # Load model and preprocessing objects
            self.model = tf.keras.models.load_model(self.model_save_path)
            self.scaler = joblib.load(self.scaler_save_path)
            self.label_encoder = joblib.load(self.encoder_save_path)
            self.feature_names = joblib.load(self.feature_names_path)

            self.is_trained = True
            print("✅ Model loaded successfully!")
            print(f"🌱 Available crops: {list(self.label_encoder.classes_)}")
            print(f"🔢 Total crops: {len(self.label_encoder.classes_)}")
            print(f"🔧 Features used: {len(self.feature_names)}")
            print(f"📋 Sample features: {self.feature_names[:5]}...")
            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return False

    def prepare_input_features(self, input_data):
        """Prepare input features for prediction"""
        try:
            # Create a DataFrame with zeros for all expected features
            input_df = pd.DataFrame(0, index=[0], columns=self.feature_names)

            # Map common input field names to actual feature names
            feature_mapping = {
                # Basic nutrients
                'N': 'N', 'P': 'P', 'K': 'K',
                # Weather and climate
                'temperature': 'temperature', 'humidity': 'humidity',
                'ph': 'ph', 'rainfall': 'rainfall',
                # Soil properties
                'Soil_Organic_Carbon_pct': 'Soil_Organic_Carbon_pct',
                'Soil_Salinity_ds_m': 'Soil_Salinity_ds_m',
                'Soil_Moisture_pct': 'Soil_Moisture_pct',
                # Location
                'Elevation_m': 'Elevation_m',
                # Historical data
                'Avg_Temp_Last_3_Months_C': 'Avg_Temp_Last_3_Months_C',
                'Avg_Rainfall_Last_3_Months_mm': 'Avg_Rainfall_Last_3_Months_mm',
                'Wind_Speed_km_h': 'Wind_Speed_km_h',
                # Economic factors
                'Market_Price_Rs_per_kg': 'Market_Price_Rs_per_kg',
                'Profitability_Index': 'Profitability_Index',
                # Sustainability
                'Water_Requirement_Index': 'Water_Requirement_Index',
                'Sustainability_Score': 'Sustainability_Score',
                # Categorical fields that might be one-hot encoded
                'Soil_Type': 'Soil_Type',
                'Climate_Zone': 'Climate_Zone',
                'Irrigation_Availability': 'Irrigation_Availability',
                'Previous_Crop': 'Previous_Crop'
            }

            # Fill in the provided values
            for input_key, feature_name in feature_mapping.items():
                if input_key in input_data and input_data[input_key] is not None:
                    # Check if the feature exists in our feature names
                    if feature_name in self.feature_names:
                        input_df[feature_name] = input_data[input_key]
                    else:
                        # For one-hot encoded features, we need to handle them differently
                        # Look for columns that start with the feature name
                        matching_features = [f for f in self.feature_names if f.startswith(feature_name + '_')]
                        if matching_features:
                            # This is a one-hot encoded feature, set the specific column
                            specific_feature = f"{feature_name}_{input_data[input_key]}"
                            if specific_feature in self.feature_names:
                                input_df[specific_feature] = 1

            # Handle missing values - set defaults for critical features
            default_values = {
                'N': 50, 'P': 50, 'K': 50,
                'temperature': 25, 'humidity': 60, 'ph': 6.5, 'rainfall': 100,
                'Soil_Organic_Carbon_pct': 1.5, 'Soil_Salinity_ds_m': 2.0,
                'Soil_Moisture_pct': 30, 'Elevation_m': 200,
                'Avg_Temp_Last_3_Months_C': 25, 'Avg_Rainfall_Last_3_Months_mm': 100,
                'Wind_Speed_km_h': 10, 'Market_Price_Rs_per_kg': 40,
                'Water_Requirement_Index': 1.5, 'Profitability_Index': 0.3,
                'Sustainability_Score': 0.5
            }

            for feature, default_value in default_values.items():
                if feature in self.feature_names and input_df[feature].iloc[0] == 0:
                    input_df[feature] = default_value

            # Ensure correct column order and fill any remaining NaN values
            input_df = input_df[self.feature_names].fillna(0)

            print(f"🔧 Prepared input with {len(input_df.columns)} features")
            print(
                f"📊 Input sample - N: {input_df['N'].iloc[0]}, P: {input_df['P'].iloc[0]}, K: {input_df['K'].iloc[0]}")
            return input_df

        except Exception as e:
            logging.error(f"❌ Error preparing input features: {str(e)}")
            raise e

    def predict_crop(self, input_data):
        """Predict suitable crops with probabilities"""
        try:
            if not self.is_trained:
                return {'error': 'Model not loaded. Please train the model first.', 'success': False}

            print(f"🎯 Making prediction with {len(input_data)} input parameters...")
            print(f"📋 Input keys: {list(input_data.keys())}")

            # Prepare input features
            input_df = self.prepare_input_features(input_data)

            # Scale the input
            input_scaled = self.scaler.transform(input_df)

            # Make prediction
            probabilities = self.model.predict(input_scaled, verbose=0)[0]
            predicted_class = np.argmax(probabilities)
            confidence = np.max(probabilities)

            # Get top recommendations
            crop_probabilities = list(zip(self.label_encoder.classes_, probabilities))
            crop_probabilities.sort(key=lambda x: x[1], reverse=True)
            top_recommendations = crop_probabilities[:5]

            predicted_crop = self.label_encoder.inverse_transform([predicted_class])[0]

            response = {
                'success': True,
                'top_recommendation': predicted_crop,
                'top_confidence': float(confidence),
                'recommendations': [
                    {
                        'crop': crop,
                        'probability': float(prob),
                        'confidence': 'High' if prob > 0.7 else 'Medium' if prob > 0.5 else 'Low'
                    }
                    for crop, prob in top_recommendations
                ],
                'all_crops_count': len(self.label_encoder.classes_),
                'features_used': len(self.feature_names),
                'input_features_received': len([k for k in input_data.keys() if input_data[k] is not None])
            }

            print(f"✅ Prediction successful: {predicted_crop} (confidence: {confidence:.3f})")
            return response

        except Exception as e:
            logging.error(f"❌ Prediction error: {str(e)}")
            import traceback
            print(f"🔍 Prediction error details: {traceback.format_exc()}")
            return {'error': f'Prediction failed: {str(e)}', 'success': False}


# Initialize the system
crop_system = CropRecommendationSystem()


def load_model_on_startup():
    """Load pre-trained model when the application starts"""
    print("=" * 60)
    print("🔄 LOADING PRE-TRAINED MODEL")
    print("=" * 60)

    print("📁 Current working directory:", os.getcwd())

    # Check if saved_models directory exists
    if os.path.exists('saved_models'):
        print("📁 Contents of saved_models directory:")
        for item in os.listdir('saved_models'):
            print(f"   ✅ {item}")
    else:
        print("❌ saved_models directory not found!")
        print("💡 Please run: python crop_recommendation_ann.py")
        return

    success = crop_system.load_trained_model()
    if success:
        print("🎉 MODEL READY FOR PREDICTIONS!")
    else:
        print("❌ MODEL FAILED TO LOAD!")
        print("💡 Please run: python crop_recommendation_ann.py")

    print("=" * 60)


# Load model on startup
load_model_on_startup()


# Serve frontend files
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": crop_system.is_trained,
        "available_crops": list(crop_system.label_encoder.classes_) if crop_system.is_trained else [],
        "total_crops": len(crop_system.label_encoder.classes_) if crop_system.is_trained else 0,
        "features_count": len(crop_system.feature_names) if crop_system.is_trained else 0,
        "message": "Model is ready for predictions!" if crop_system.is_trained else "Please train the model first"
    })


@app.route('/api/predict', methods=['POST'])
def predict_crop():
    try:
        # Check if model is loaded first
        if not crop_system.is_trained:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first by running: python crop_recommendation_ann.py'
            }), 400

        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No input data provided'}), 400

        logging.info(f"📥 Received prediction request with {len(data)} parameters")
        print(f"🔍 Prediction input data: {data}")

        prediction = crop_system.predict_crop(data)

        return jsonify(prediction)

    except Exception as e:
        logging.error(f"❌ Prediction endpoint error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/crop_stats', methods=['GET'])
def get_crop_stats():
    if not crop_system.is_trained:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 400

    all_crops = list(crop_system.label_encoder.classes_)

    # Categorize crops based on typical characteristics
    high_yield = ['rice', 'maize', 'wheat', 'sugarcane']
    high_profit = ['pomegranate', 'grapes', 'apple', 'orange', 'mango']
    sustainable = ['chickpea', 'lentil', 'pigeonpeas', 'mothbeans', 'mungbean']
    drought_resistant = ['pigeonpeas', 'mothbeans', 'sorghum', 'millet']
    water_loving = ['rice', 'sugarcane', 'jute', 'banana']

    stats = {
        'success': True,
        'high_yield_crops': [crop for crop in high_yield if crop in all_crops],
        'high_profit_crops': [crop for crop in high_profit if crop in all_crops],
        'sustainable_crops': [crop for crop in sustainable if crop in all_crops],
        'drought_resistant': [crop for crop in drought_resistant if crop in all_crops],
        'water_loving': [crop for crop in water_loving if crop in all_crops],
        'all_crops': all_crops,
        'total_crops': len(all_crops),
        'model_loaded': True
    }
    return jsonify(stats)


@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    if not crop_system.is_trained:
        return jsonify({
            'success': False,
            'is_loaded': False,
            'message': 'Model not loaded. Please run the training script first.'
        })

    return jsonify({
        'success': True,
        'is_loaded': True,
        'available_crops': list(crop_system.label_encoder.classes_),
        'total_crops': len(crop_system.label_encoder.classes_),
        'model_type': 'Advanced ANN (Keras)',
        'features_count': len(crop_system.feature_names),
        'sample_features': crop_system.feature_names[:10],  # Show first 10 features
        'message': 'Model is ready for predictions!'
    })


@app.route('/api/fertilizer_recommendation', methods=['POST'])
def recommend_fertilizer():
    try:
        data = request.json
        n = data.get('N', data.get('n', 0))
        p = data.get('P', data.get('p', 0))
        k = data.get('K', data.get('k', 0))
        ph = data.get('pH', data.get('ph', 6.5))
        soil_organic = data.get('Soil_Organic_Carbon_pct', data.get('soil_organic_carbon', 1.5))

        # Improved fertilizer logic based on your dataset
        if n < 25 and p < 20 and k < 20:
            fertilizer = "Complete NPK (10-26-26)"
            dosage = "200-250 kg/ha"
            reason = "Low levels of all major nutrients"
        elif n < 30:
            fertilizer = "Urea (46-0-0)"
            dosage = "150-200 kg/ha"
            reason = "Nitrogen deficiency detected"
        elif p < 25:
            fertilizer = "DAP (18-46-0)"
            dosage = "120-150 kg/ha"
            reason = "Phosphorus deficiency detected"
        elif k < 25:
            fertilizer = "MOP (0-0-60)"
            dosage = "80-120 kg/ha"
            reason = "Potassium deficiency detected"
        elif ph < 5.5:
            fertilizer = "Lime + Complex NPK"
            dosage = "2-4 tons/ha lime + 150 kg/ha NPK"
            reason = "Soil is too acidic"
        elif ph > 8.0:
            fertilizer = "Gypsum + Complex NPK"
            dosage = "2-3 tons/ha gypsum + 150 kg/ha NPK"
            reason = "Soil is too alkaline"
        elif soil_organic < 1.0:
            fertilizer = "Organic Manure + Complex NPK"
            dosage = "8-12 tons/ha manure + 120 kg/ha NPK"
            reason = "Low organic matter content"
        else:
            fertilizer = "Complex NPK (14-35-14)"
            dosage = "150-180 kg/ha"
            reason = "Balanced nutrient maintenance"

        return jsonify({
            'success': True,
            'recommended_fertilizer': fertilizer,
            'dosage': dosage,
            'reason': reason,
            'soil_analysis': {
                'pH': f"{ph:.1f}",
                'NPK_ratio': f"{n}-{p}-{k}",
                'organic_carbon': f"{soil_organic:.1f}%"
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/check_model', methods=['GET'])
def check_model():
    """Special endpoint to check model status in detail"""
    model_files = {
        'model_file': os.path.exists('saved_models/crop_recommendation_model.h5'),
        'scaler_file': os.path.exists('saved_models/scaler.pkl'),
        'encoder_file': os.path.exists('saved_models/label_encoder.pkl'),
        'features_file': os.path.exists('saved_models/feature_names.pkl')
    }

    all_files_exist = all(model_files.values())

    return jsonify({
        'model_loaded': crop_system.is_trained,
        'files_exist': model_files,
        'all_files_present': all_files_exist,
        'current_directory': os.getcwd(),
        'saved_models_exists': os.path.exists('saved_models'),
        'message': 'Model is ready!' if all_files_exist else 'Some files are missing!'
    })


@app.route('/api/reload_model', methods=['POST'])
def reload_model():
    """Endpoint to reload the model without restarting the server"""
    try:
        print("🔄 Manually reloading model...")
        success = crop_system.load_trained_model()
        if success:
            return jsonify({
                'success': True,
                'message': 'Model reloaded successfully!',
                'model_loaded': True
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to reload model',
                'model_loaded': False
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error reloading model: {str(e)}',
            'model_loaded': False
        })


def open_browser():
    """Open browser after a short delay"""
    time.sleep(3)
    webbrowser.open('http://localhost:5000')


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 STARTING CROP RECOMMENDATION SYSTEM")
    print("=" * 60)

    print("📊 Model Status:", "✅ LOADED AND READY" if crop_system.is_trained else "❌ NOT LOADED")

    if crop_system.is_trained:
        print("🎯 Total Available Crops:", len(crop_system.label_encoder.classes_))
        print("🌱 Crops:", list(crop_system.label_encoder.classes_))
        print("🔧 Features Used:", len(crop_system.feature_names))
        print("💡 You can now make predictions!")
    else:
        print("❌ MODEL NOT LOADED - Predictions will not work!")
        print("💡 Solution: Run 'python crop_recommendation_ann.py' first")
        print("💡 Then restart this Flask app")

    print("🌐 Starting Flask server on http://localhost:5000")
    print("📱 Frontend will open automatically in your browser...")
    print("=" * 60)

    # Open browser automatically
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)