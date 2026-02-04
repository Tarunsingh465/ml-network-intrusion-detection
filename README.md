## 🔐 ML-Based Network Intrusion Detection System (NIDS)

A Machine Learning–based Network Intrusion Detection System built on the CICIDS 2017 dataset, designed to detect malicious network traffic, visualize attack patterns, and explain model decisions in an interpretable manner.

This project integrates Machine Learning + Cyber Security with a full backend–frontend architecture, making it suitable for academic evaluation, viva, and interview discussions.


## 📌 Project Highlights

1. Binary intrusion detection (BENIGN vs ATTACK)
2. Probability-based prediction with confidence scores
3. Batch CSV analysis for large network traffic logs
4. Interactive dashboard with real-time visualizations
5. Explainability module for model transparency
6. Prediction history tracking for reproducibility


## 🧠 Problem Statement

Traditional signature-based intrusion detection systems fail to detect modern and zero-day attacks.
This project addresses that limitation by using Machine Learning models trained on real network traffic to automatically identify malicious behavior.


## 📊 Dataset

CICIDS 2017:

1. Realistic network traffic dataset
2. Includes both benign and multiple attack scenarios
3. High-dimensional flow-based features

Preprocessing steps include:
1. Removal of label leakage
2. Handling missing and infinite values
3. Feature alignment with training schema


## ⚙️ System Architecture:
User (Browser)
   ↓
Streamlit Dashboard (Frontend)
   ↓  REST API
Flask Backend
   ↓
Trained Random Forest Model
   ↓
Predictions + Confidence + History Logs


## 🧪 Machine Learning Model

Algorithm: Random Forest Classifier
Why Random Forest?
Handles high-dimensional data well
Robust to noise
Provides built-in feature importance (explainability)

Output per flow:
Prediction label (BENIGN / ATTACK)
Attack confidence score
Benign confidence score

A configurable probability threshold is used to balance sensitivity and false positives.


## 📈 Dashboard Features:

1️⃣ Single Flow Prediction
Manual feature input
Real-time prediction
Confidence visualization
Risk level interpretation

2️⃣ Batch CSV Analysis:
Upload CICIDS-style CSV files
Predict thousands of flows at once

Displays:
Total flows
Benign vs attack counts
Traffic distribution graphs

3️⃣ Visualization Module:
Pie chart: attack vs benign distribution
Bar chart: comparative traffic volume
Feature impact trends

4️⃣ Explainability Module:
Global feature importance from the trained model
Dataset-aware contextual scaling
Attack vs benign feature influence comparison
Human-readable explanation for non-technical users

⚠️ Note: Explainability reflects global model behavior, not per-flow SHAP values.

5️⃣ Prediction History:

Logs every batch prediction
Stores:
Timestamp
CSV filename
Threshold
Total, benign, and attack counts
Enables run-level traceability


## 🔍 Explainability Design (Important):
The explainability module uses global feature importance derived from the Random Forest model.
This remains consistent across datasets, while dataset context (attack ratio) is used to visually emphasize differences between runs.

This design ensures:
ML correctness
Interpretability
Viva-safe explanations

## 🛠️ Tech Stack:
Python
Scikit-learn
Pandas / NumPy
Flask (Backend API)
Streamlit (Frontend Dashboard)
Plotly (Interactive Visualizations)


▶️ How to Run the Project:

1️⃣ Create virtual environment (optional)
python -m venv .venv


2️⃣ Install dependencies:
pip install -r requirements.txt


3️⃣ Start backend server:
python app.py


4️⃣ Start dashboard:
streamlit run dashboard.py

