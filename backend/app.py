from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime


# ==============================
# BASIC SETUP
# ==============================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "random_forest_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "model", "feature_columns.pkl")
LOG_PATH = os.path.join(BASE_DIR, "..", "logs", "history_logs.csv")


# ==============================
# LOAD MODEL & FEATURES
# ==============================
model = joblib.load(MODEL_PATH)

# feature list used during training
FEATURE_COLUMNS = joblib.load(FEATURE_PATH)

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return "Flask backend is running"

@app.route("/ping")
def ping():
    return {"status": "Flask is alive"}

# ------------------------------
# SINGLE FLOW PREDICTION
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)

    proba = model.predict_proba(features)[0]

    benign_confidence = float(proba[0])
    attack_confidence = float(proba[1])

    prediction = int(attack_confidence > 0.3)

    return jsonify({
        "prediction": prediction,
        "label": "ATTACK" if prediction == 1 else "BENIGN",
        "benign_confidence": benign_confidence,
        "attack_confidence": attack_confidence
    })


# ------------------------------
# SAVE HISTORY
# ------------------------------
def save_history(csv_name, threshold, total, benign, attack):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "csv_name": csv_name,
        "threshold": threshold,
        "total_flows": total,
        "benign_count": benign,
        "attack_count": attack
    }

    df_row = pd.DataFrame([row])

    if os.path.exists(LOG_PATH):
        df_row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(LOG_PATH, index=False)

# ------------------------------
# BATCH CSV PREDICTION
# ------------------------------
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        threshold = float(request.args.get("threshold", 0.1))
        file = request.files["file"]
        df = pd.read_csv(file)

        if "Label" in df.columns:
            df = df.drop(columns=["Label"])

        df = df.select_dtypes(include=["number"])
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        df = df[FEATURE_COLUMNS]

        probs = model.predict_proba(df)[:, 1]
        preds = (probs > threshold).astype(int)

        total_flows = int(len(preds))
        normal_count = int((preds == 0).sum())
        attack_count = int((preds == 1).sum())

        # ✅ THIS WAS MISSING (HISTORY SAVE)
        save_history(
            csv_name=file.filename,
            threshold=threshold,
            total=total_flows,
            benign=normal_count,
            attack=attack_count
        )

        return jsonify({
            "total_flows": total_flows,
            "normal_count": normal_count,
            "attack_count": attack_count
        })

    except Exception as e:
        print("BACKEND ERROR:", e)
        return jsonify({"error": "failed"}), 500


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
