import requests
import json


API_URL = "http://127.0.0.1:5000/predict"

# Sample input data
payload = {
    "area_sqft": 10000,
    "location": "dha phase 6",
    "property_type": "house",
    "bedrooms": 6,
    "province_name": "punjab"
}

headers = {
    "Content-Type": "application/json"
}

try:
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction Result:")
        print("Predicted Price:", result.get('final_ensemble_price'))
        print("Investment Advice:", result.get('investment_advice'))
        if result.get('note'):
            print("Note:", result['note'])

        recommendations = result.get('recommendations', [])
        if recommendations:
            print("\n💡 Random Location Suggestions (based on your input):")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n  [{i}] 📍 Location   : {rec['location']}")
                print(f"       🏙️  City       : {rec.get('city', 'N/A')}")
                print(f"       💰 Price      : {rec['price']:,}")
                print(f"       📐 Area       : {rec['area_sqft']:,} sqft")
                print(f"       📊 Price/Sqft : {rec['price_per_sqft']}")
        else:
            print("No alternative recommendations found.")
    else:
        print("❌ Error:", response.status_code, response.text)

except Exception as e:
    print("❌ Exception occurred while making the request:", str(e))
