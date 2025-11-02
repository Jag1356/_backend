from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.pkl"
DATA_PATH = "synthetic_cancer_data.csv"

# Train and save model if not already saved
def train_and_save_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["predicted_cost"])
    y = df["predicted_cost"]

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X)
    columns = X_encoded.columns

    # Train model
    X_train, _, y_train, _ = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and columns
    joblib.dump({"model": model, "columns": list(columns)}, MODEL_PATH)
    print("✅ Model trained and saved.")

# Load model if exists, else train it
def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return joblib.load(MODEL_PATH)

# Load model at startup
model_data = load_model()

# Preprocess incoming input
def preprocess_input(data):
    df = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df)
    return df_encoded.reindex(columns=model_data["columns"], fill_value=0)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        required_fields = {"age", "gender", "cancer_type", "cancer_stage"}

        if not required_fields.issubset(data.keys()):
            return jsonify({"error": "Missing input fields"}), 400

        processed = preprocess_input(data)
        prediction = model_data["model"].predict(processed)[0]

        return jsonify({"predicted_cost": round(prediction, 2)})

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)




# how to Run
# cd frontend nd backend side by side
# in frontend npm i nd npm run dev 
# in backend python app.PythonFinalizationErrorenter
