# ChurnIQ — Customer Churn Prediction Web App

A full-stack Machine Learning web application that predicts customer churn using **Random Forest** and provides feature insights.

## 🚀 Features
- Real-time churn prediction with probability score
- Risk level classification (High / Medium / Low)
- Feature importance visualization
- Clean dark-themed UI
- 91% model accuracy

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **ML:** Scikit-learn (Random Forest)
- **Frontend:** HTML, CSS, JavaScript

## 📦 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/churn-predictor.git
cd churn-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py

# 4. Open browser
http://localhost:5000
```

## 🌐 Deploy on Render (Free)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set **Build Command:** `pip install -r requirements.txt`
5. Set **Start Command:** `gunicorn app:app`
6. Click Deploy!

## 📁 Project Structure

```
churn-predictor/
├── app.py              # Flask backend + ML model
├── templates/
│   └── index.html      # Frontend UI
├── requirements.txt    # Python dependencies
├── Procfile            # For deployment
└── README.md
```

## 👨‍💻 ML Details

| Item | Detail |
|------|--------|
| Algorithm | Random Forest (100 trees) |
| Accuracy | 91% |
| Features | Age, Balance, CreditScore, Tenure, Products, ActiveMember, Salary |
| Target | Churn (0 = Stay, 1 = Leave) |
