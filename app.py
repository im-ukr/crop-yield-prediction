from flask import Flask, request, jsonify, render_template,send_file
import joblib
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import pickle
import xgboost as xgb

app = Flask(__name__, static_folder="static", template_folder=".")

# ================== Common Data Loading ==================
df = pd.read_csv("datasets/crop_production.csv")
df_clean = df.copy()
df_clean["State_Name"] = df_clean["State_Name"].str.strip()
df_clean["District_Name"] = df_clean["District_Name"].str.strip()
df_clean["Season"] = df_clean["Season"].str.strip()
df_clean["Crop"] = df_clean["Crop"].str.strip()

# ================== Yield Prediction Setup ==================
yield_model = joblib.load("models/crop_production_model.joblib")
state_district_yield = df_clean.groupby('State_Name')['District_Name'].apply(set).to_dict()

# Create label encoders for yield model
yield_encoders = {
    "State": LabelEncoder().fit(df_clean["State_Name"].unique()),
    "District": LabelEncoder().fit(df_clean["District_Name"].unique()),
    "Season": LabelEncoder().fit(df_clean["Season"].unique()),
    "Crop": LabelEncoder().fit(df_clean["Crop"].unique())
}

# ================== Crop Recommendation Setup ==================
# recommend_model = joblib.load("models/xgboost_production_model.joblib")
recommend_model = xgb.XGBClassifier()
recommend_model.load_model("models/xgboost_classifier_new_new.json")
recommend_encoders=joblib.load("models/label_encoders_new_new.joblib")




# Lowercase validation data
df_lower = df_clean.copy()
df_lower["State_Name"] = df_lower["State_Name"].str.lower()
df_lower["District_Name"] = df_lower["District_Name"].str.lower()
state_district_recommend = df_lower.groupby('State_Name')['District_Name'].apply(set).to_dict()

# Recommendation encoders
state_to_code = {state: idx for idx, state in enumerate(sorted(df_lower["State_Name"].unique()))}
district_to_code = {district: idx for idx, district in enumerate(sorted(df_lower["District_Name"].unique()))}


# Create a dictionary mapping crops to PDF paths
CROP_GUIDES = {
    "Rice": "static/pdf/rice.pdf",
    "Wheat": "static/pdf/wheat.pdf",
    "Maize": "static/pdf/maize.pdf",
    "Apple":"static/pdf/apple.pdf",
    "Bajra":"static/pdf/bajra.pdf",
    "Banana":"static/pdf/banana.pdf",
    "Sugarcane":"static/pdf/sugarcane.pdf",
    "Urad":"static/pdf/urad.pdf",
    "Yam":"static/pdf/yam.pdf",
    "Paddy":"static/pdf/paddy.pdf",
    "Sunflower":"static/pdf/sunflower.pdf",
    "Beet Root":"static/pdf/beetroot.pdf",
    "Sweet potato":"static/pdf/sweetpotatoe.pdf",
    "Turmeric":"static/pdf/turmeric.pdf",
    "Ragi":"static/pdf/ragi.pdf",
    "Potato":"static/pdf/potato.pdf",
    "Dry chillies":"static/pdf/drychilli.pdf",
    "Ginger":"static/pdf/ginger.pdf",
    "Arecanut":"static/pdf/arecanut.pdf",
    "Soyabean":"static/pdf/soybean.pdf",
    "Sesamum":"static/pdf/sesamum.pdf",
    "Litchi":"static/pdf/litchi.pdf",
    "Garlic":"static/pdf/garlic.pdf",
    "Dry ginger":"static/pdf/dryginger.pdf",
    "Ber":"static/pdf/ber.pdf",
    "Arhar/Tur":"static/pdf/arhar.pdf",
    "Barley":"static/pdf/barley.pdf",
    "Cabbage":"static/pdf/cabbage.pdf",
    "Onion":"static/pdf/onion.pdf",
    "Ash Gourd":"static/pdf/ashgourd.pdf",
    "Tobacco":"static/pdf/tobacco.pdf",
    "Coconut":"static/pdf/coconut.pdf",
    "Castor seed":"static/pdf/Castorseed.pdf",
    "Tomato":"static/pdf/tomato.pdf",
    "Coffe":"static/pdf/coffe.pdf",
    "Tea":"static/pdf/tea.pdf",
    "Jute":"static/pdf/jute.pdf",
    "Mesta":"static/pdf/mesta.pdf",
    "Jowar":"static/pdf/jowar.pdf",
    "Mango":"static/pdf/mango.pdf",
    "Water Melon":"static/pdf/watermelon.pdf",
    "Jute & mesta":"static/pdf/jutemesta.pdf",
    "Gram":"static/pdf/gram.pdf",
    "Blackgram":"static/pdf/blackgram.pdf",
    "Horse-gram":"static/pdf/horsegram.pdf",
    "Moong(Green Gram)":"static/pdf/moong.pdf",
    "Cotton(lint)":"static/pdf/cotton.pdf",
    "Khesari":"static/pdf/khesari.pdf",
    "Orange":"static/pdf/orange.pdf",
    "Brinjal":"static/pdf/brinjal.pdf",
    "Grapes":"static/pdf/grape.pdf",
    "Carrot":"static/pdf/carrot.pdf",
    "Rubber":"static/pdf/rubber.pdf",
    "Turnip":"static/pdf/turnip.pdf",
    "Small millets":"static/pdf/small millets.pdf",
    "Cauliflower":"static/pdf/cauiliflower.pdf",
    "Safflower":"static/pdf/safflower.pdf",
    "Papaya":"static/pdf/papaya.pdf",
    "Bean":"static/pdf/bean.pdf",
    "Pump Kin":"static/pdf/pumpkin.pdf",
    "Lentil":"static/pdf/lentil.pdf",
    "Oilseeds total":"static/pdf/oilseeds.pdf",
    "other oilseeds":"static/pdf/oilseeds2.pdf",
    "Bitter Gourd":"static/pdf/bittergourd.pdf",
    "Bottle Gourd":"static/pdf/bottlegourd.pdf",
    "Cucumber":"static/pdf/cucumber.pdf",
    "Cashewnut":"static/pdf/kaju.pdf",
    "Cashewnut Processed":"static/pdf/kaju2.pdf",
    "Ribed Guard":"static/pdf/ribedgourd.pdf",
    "Drum Stick":"static/pdf/drumstick.pdf",
    "Redish":"static/pdf/radish.pdf",
    "Guar seed":"static/pdf/gaurseed.pdf",
    "Groundnut":"static/pdf/groundnut.pdf",
    "Masoor":"static/pdf/Masoor.pdf",
    "Pulses total":"static/pdf/pulses.pdf",
    "Other Vegetables":"static/pdf/vegetables.pdf",
    "Linseed":"static/pdf/linseed.pdf",
    "Niger seed":"static/pdf/nigerseed.pdf",
    "Pear":"static/pdf/pear.pdf",
    "Pineapple":"static/pdf/pineapple.pdf",
    "Plums":"static/pdf/plum.pdf",
    "Bhindi":"static/pdf/bhindi.pdf",
    "Cowpea(Lobia)":"static/pdf/cowpea.pdf",
    "Black pepper":"static/pdf/blackpepper.pdf",
    "Arcanut (Processed)":"static/pdf/Arcanutprocessed.pdf",
    "other fibres":"static/pdf/fibre.pdf",
    "Other Kharif pulses":"static/pdf/kharifpulse.pdf",
    "Other Kharif pulses":"static/pdf/kharifpulse.pdf",
    "Other  Rabi pulses":"static/pdf/rabipulse.pdf",
    "Peas  (vegetable)":"static/pdf/peas.pdf",

}

@app.route('/download-strategy')
def download_strategy():
    crop = request.args.get('crop', '')
    file_path = CROP_GUIDES.get(crop.strip(), None)
    
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "Strategy not found"}), 404
        
    return send_file(file_path, as_attachment=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model')
def model_page():
    return render_template('model.html',
        states=json.dumps(sorted(df_clean["State_Name"].unique().tolist())),
        districts=json.dumps(sorted(df_clean["District_Name"].unique().tolist())),
        seasons=json.dumps(sorted(df_clean["Season"].unique().tolist())),
        crops=json.dumps(sorted(df_clean["Crop"].unique().tolist()))
    )

@app.route('/autocomplete/<field>')
def autocomplete(field):
    query = request.args.get('query', '').lower()
    suggestions = []
    
    if field == 'state':
        suggestions = [s for s in df_clean["State_Name"].unique() if query in s.lower()]
    elif field == 'district':
        suggestions = [d for d in df_clean["District_Name"].unique() if query in d.lower()]
    elif field == 'season':
        suggestions = [s for s in df_clean["Season"].unique() if query in s.lower()]
    elif field == 'crop':
        suggestions = [c for c in df_clean["Crop"].unique() if query in c.lower()]
    
    return jsonify(suggestions[:15])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = {"yield": None, "crop": None, "error": None}

    try:
        state = data.get("State_Name", "").strip()
        district = data.get("District_Name", "").strip()
        
        # Validate state and district
        valid_districts = state_district_yield.get(state, set())
        valid_districts_lower = state_district_recommend.get(state.lower(), set())
        
        if not state or not district:
            result["error"] = "State and District are required fields"
        elif district not in valid_districts or district.lower() not in valid_districts_lower:
            result["error"] = f"Invalid district for {state}. Valid districts: {', '.join(sorted(valid_districts))}"
        else:
            # ================== Yield Prediction ==================
            try:
                # Create DataFrame for yield model
                input_yield = pd.DataFrame([{
                    "State_Name": state,
                    "District_Name": district,
                    "Season": data.get("Season", "").strip(),
                    "Crop": data.get("Crop", "").strip(),
                    "Area": float(data.get("Area", 0)),
                    "Crop_Year": float(data.get("Crop_Year", 0))
                }])

                # Arrange columns in correct order
                input_yield = input_yield[yield_model.feature_names_in_]

                # Predict yield
                yield_prediction = yield_model.predict(input_yield)[0]
                result["yield"] = f"{yield_prediction:.2f} Tons"

            except Exception as e:
                result["error"] = f"Yield prediction error: {str(e)}"

            # ================== Crop Recommendation ==================
            try:
                if not result.get("error"):
                    # Prepare DataFrame for recommendation
                    input_recommend = pd.DataFrame([{
                        'State_Name': state,
                        'District_Name': district,
                        'Crop_Year': float(data.get("Crop_Year", 0)),
                        'Season': data.get("Season", "").strip(),
                        'Area': float(data.get("Area", 0))
                    }])

                    # Encode categorical columns
                    for col in ['State_Name', 'District_Name', 'Season']:
                        le = recommend_encoders[col]
                        input_recommend[col] = input_recommend[col].apply(
                            lambda x: x if x in le.classes_ else le.classes_[0]
                        )
                        input_recommend[col] = le.transform(input_recommend[col])

                    # Predict crop
                    crop_pred = recommend_model.predict(input_recommend)[0]
                    result["crop"] = recommend_encoders["Crop"].inverse_transform([crop_pred])[0]

            except Exception as e:
                if not result.get("yield"):
                    result["error"] = f"Crop recommendation error: {str(e)}"

    except Exception as e:
        result["error"] = str(e)

    return jsonify(result)




## Experiments
# Load model and encoders
global_model = joblib.load("models/random_forest_model.joblib")
label_encoders = joblib.load("models/latest_label_encoders.joblib")


@app.route('/harvest')
def harvest():
    return render_template('harvest.html')

# Define feature order expected by the model
FEATURE_ORDER = [
    'Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
    'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition', 'Days_to_Harvest'
]

@app.route('/predict-harvest', methods=['POST'])
def predict_harvest():
    try:
        data = request.json
        
        # Validate and convert inputs
        def validate_numeric(value, field, min_val, max_val):
            try:
                num = float(value)
                if not (min_val <= num <= max_val):
                    raise ValueError
                return num
            except (ValueError, TypeError):
                raise ValueError(f"Invalid {field}. Must be between {min_val}-{max_val}")

        # Create input dictionary
        raw_data = {
            "Region": data.get("region", "").strip(),
            "Soil_Type": data.get("soilType", "").strip(),
            "Crop": data.get("crop", "").strip(),
            "Rainfall_mm": validate_numeric(data.get("rainfall"), "rainfall", 0, 3000),
            "Temperature_Celsius": validate_numeric(data.get("temperature"), "temperature", -50, 60),
            "Fertilizer_Used": int(data.get("fertilizer", 0)),
            "Irrigation_Used": int(data.get("irrigation", 0)),
            "Weather_Condition": data.get("weather", "").strip(),
            "Days_to_Harvest": validate_numeric(data.get("daysToHarvest"), "daysToHarvest", 1, 365)
        }

        # Create DataFrame
        input_df = pd.DataFrame([raw_data], columns=FEATURE_ORDER)

        # Encode categorical features
        encoded_data = input_df.copy()
        for col in ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']:
            # Handle unseen categories
            encoded_data[col] = encoded_data[col].apply(
                lambda x: x if x in label_encoders[col].classes_ else 'unknown'
            )
            encoded_data[col] = label_encoders[col].transform(encoded_data[col])

        # Convert to numpy array
        processed_input = encoded_data.values.reshape(1, -1)

        # Make prediction
        prediction = global_model.predict(processed_input)[0]
        
        return jsonify({
            "yield": f"{prediction:.2f} Tons",
            "error": None
        })

    except Exception as e:
        return jsonify({
            "yield": None,
            "error": f"Prediction error: {str(e)}"
        }), 400


if __name__ == '__main__':
    app.run(debug=True)
