from flask import Flask, request, jsonify
import pandas as pd
import joblib
from difflib import get_close_matches
import os
import numpy as np

app = Flask(__name__)


#  MODEL AND DATA LOADING

try:
    model = joblib.load('model/xgb_land_price_model.joblib')
    objects = joblib.load('model/xgb_preprocessing.joblib')
    encoders = objects['encoders']
    scaler = objects['scaler']

    df_original = pd.read_csv("DataSet/Property_with_Feature_Engineering.csv")
    df_original.dropna(
        subset=['area_sqft', 'location', 'property_type', 'bedrooms', 'province_name', 'price'],
        inplace=True
    )
    df_original['location'] = df_original['location'].astype(str).str.strip().str.lower()
    df_original['property_type'] = df_original['property_type'].astype(str).str.strip().str.lower()
    df_original['province_name'] = df_original['province_name'].astype(str).str.strip().str.lower()
except Exception as e:
    raise RuntimeError(f"❌ Error loading model or dataset: {e}")



#  ROUTES

@app.route('/')
def index():
    return jsonify({"message": "✅ Land Price Prediction API is running"}), 200


@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200



#  PREDICTION ENDPOINT

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        required_fields = ['area_sqft', 'location', 'property_type', 'bedrooms', 'province_name']
        for f in required_fields:
            if f not in data:
                return jsonify({'error': f"Missing field: {f}"}), 400

        # --- Clean Inputs ---
        area = float(data['area_sqft'])
        location = str(data['location']).strip().lower()
        property_type = str(data['property_type']).strip().lower()
        bedrooms = int(data['bedrooms'])
        province = str(data['province_name']).strip().lower()

        # --- Fuzzy Match Province ---
        known_provinces = df_original['province_name'].unique().tolist()
        province_match = get_close_matches(province, known_provinces, n=1, cutoff=0.6)
        province = province_match[0] if province_match else province

        # --- Fuzzy Match Location ---
        province_locs = df_original[df_original['province_name'] == province]['location'].unique().tolist()
        loc_match = get_close_matches(location, province_locs, n=1, cutoff=0.5)
        used_location = loc_match[0] if loc_match else location

        # --- Prepare Input DataFrame ---
        input_dict = {
            'area_sqft': area,
            'location': used_location,
            'property_type': property_type,
            'bedrooms': bedrooms,
            'province_name': province
        }
        input_df = pd.DataFrame([input_dict])

        # --- Encode Categoricals ---
        for col in ['location', 'property_type', 'province_name']:
            val = input_df.at[0, col]
            if val not in encoders[col].classes_:
                close = get_close_matches(val, encoders[col].classes_, n=1, cutoff=0.5)
                input_df.at[0, col] = close[0] if close else encoders[col].classes_[0]
            input_df[col] = encoders[col].transform(input_df[col])

        # --- Scale numeric features ---
        input_df['area_sqft'] = scaler.transform(input_df[['area_sqft']])

        # --- Predict ---
        raw_pred = float(model.predict(input_df)[0])

        # --- Reverse transformation if model used log scale ---
        if raw_pred < 20:
            pred_price = np.expm1(raw_pred)
        elif raw_pred < 1000:
            mean_price = df_original['price'].mean()
            std_price = df_original['price'].std()
            pred_price = (raw_pred * std_price) + mean_price
        else:
            pred_price = raw_pred

        # --- Investment Advice ---
        advice = "Profitable Investment " if pred_price <= 10_000_000 else "Not Profitable ❌"

        # --- Recommendations (if not profitable) ---
        recommendations = []
        if "Not Profitable" in advice:
            similar = df_original[
                (df_original['province_name'] == province) &
                (df_original['property_type'] == property_type) &
                (df_original['bedrooms'] == bedrooms)
            ].copy()

            if not similar.empty:
                similar['price_per_sqft'] = similar['price'] / similar['area_sqft']
                sample_size = min(3, len(similar))
                random_picks = similar.sample(n=sample_size, random_state=None)

                for _, row in random_picks.iterrows():
                    recommendations.append({
                        'location': row['location'],
                        'city': row.get('city', ''),
                        'price': int(row['price']),
                        'area_sqft': int(row['area_sqft']),
                        'price_per_sqft': round(row['price_per_sqft'], 2)
                    })

        #  Return clean response (no warnings or log notes)
        return jsonify({
            'final_ensemble_price': round(pred_price, 2),
            'investment_advice': advice,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



#  GET LOCATIONS BY PROVINCE

@app.route('/get_locations_by_province', methods=['GET'])
def get_locations():
    try:
        province = request.args.get('province', '').lower().strip()
        if not province:
            return jsonify({'error': 'Province is required'}), 400

        province_match = get_close_matches(province, df_original['province_name'].unique().tolist(), n=1, cutoff=0.6)
        used_province = province_match[0] if province_match else province

        filtered = df_original[df_original['province_name'] == used_province][
            ['location', 'city', 'latitude', 'longitude']
        ].dropna().drop_duplicates()

        if filtered.empty:
            return jsonify({'error': f"No locations found for province '{province}'"}), 404

        return jsonify({'province_used': used_province, 'locations': filtered.to_dict(orient='records')})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



#  GET SPECIFIC LOCATION INFO

@app.route('/get_location', methods=['GET'])
def get_location():
    try:
        province = request.args.get('province', '').lower().strip()
        location = request.args.get('location', '').lower().strip()

        if not province or not location:
            return jsonify({'error': 'Province and Location are required'}), 400

        province_df = df_original[df_original['province_name'] == province]
        if province_df.empty:
            return jsonify({'error': f"No data found for province '{province}'"}), 404

        known_locs = province_df['location'].unique().tolist()
        loc_match = get_close_matches(location, known_locs, n=1, cutoff=0.5)
        used_location = loc_match[0] if loc_match else location

        match = province_df[province_df['location'] == used_location]
        if match.empty:
            return jsonify({'error': f"No location found for '{location}' in '{province}'"}), 404

        row = match.iloc[0]
        return jsonify({
            'location': row['location'],
            'city': row.get('city', ''),
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'note': f"Matched location: {used_location}" if used_location != location else ""
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



#  APP ENTRY POINT

if __name__ == "__main__":
    port = 5000
    print(f"✅ Flask running at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
