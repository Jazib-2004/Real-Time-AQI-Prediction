from flask import Flask, request, jsonify
import xgboost as xgb

app = Flask(__name__)



# For JSON model 
model = xgb.Booster()
model.load_model("xgb_model.json")

@app.route("/predict", methods=["POST"])
def predict():
    # Receive input JSON data
    data = request.get_json()

    # Extract features from input data
    features = [data["temp"], data["humidity"], data["no2"], data["so2"]]

    # Make predictions using the loaded model
    dmatrix = xgb.DMatrix([features])
    prediction = model.predict(dmatrix)

    # Return the prediction as JSON
    return jsonify({"AQI": prediction.tolist()})

@app.route("/",methods=["GET"])
def home():
    return "Welcome to Real Time AQI Prediction"
    
if __name__ == "__main__":
    app.run(debug=True)
