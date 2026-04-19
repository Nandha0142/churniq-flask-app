from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── Train model on startup ─────────────────────────────────────────────────────
def train_model():
    np.random.seed(42)
    n = 500
    data = pd.DataFrame({
        "Age"            : np.random.randint(18, 70, n),
        "Balance"        : np.random.uniform(0, 100000, n).round(2),
        "NumProducts"    : np.random.randint(1, 5, n),
        "IsActiveMember" : np.random.randint(0, 2, n),
        "CreditScore"    : np.random.randint(300, 850, n),
        "Tenure"         : np.random.randint(0, 10, n),
        "EstimatedSalary": np.random.uniform(20000, 150000, n).round(2),
    })
    cp = (0.3*(data["Balance"]<20000) + 0.3*(data["IsActiveMember"]==0) +
          0.2*(data["CreditScore"]<500) + 0.2*(data["NumProducts"]==1))
    data["Churn"] = (cp + np.random.uniform(0, 0.3, n) > 0.5).astype(int)

    features = ["Age","Balance","NumProducts","IsActiveMember",
                "CreditScore","Tenure","EstimatedSalary"]
    X, y = data[features], data["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, round(acc * 100, 2)

model, model_accuracy = train_model()
FEATURES = ["Age","Balance","NumProducts","IsActiveMember",
            "CreditScore","Tenure","EstimatedSalary"]

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", accuracy=model_accuracy)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        sample = pd.DataFrame([{
            "Age"            : float(data["age"]),
            "Balance"        : float(data["balance"]),
            "NumProducts"    : int(data["numProducts"]),
            "IsActiveMember" : int(data["isActive"]),
            "CreditScore"    : float(data["creditScore"]),
            "Tenure"         : int(data["tenure"]),
            "EstimatedSalary": float(data["salary"]),
        }])
        pred = int(model.predict(sample)[0])
        prob = float(model.predict_proba(sample)[0][1])
        importance = dict(zip(FEATURES, model.feature_importances_.tolist()))

        # Risk level
        if prob >= 0.75:   risk = "High"
        elif prob >= 0.45: risk = "Medium"
        else:              risk = "Low"

        return jsonify({
            "prediction" : pred,
            "probability": round(prob * 100, 1),
            "risk"       : risk,
            "importance" : importance,
            "status"     : "success"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
