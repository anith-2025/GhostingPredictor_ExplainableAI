## 👻 GhostBusters

Ghosting predictor for dating apps. Uses 5 ML models + SHAP explainability.

## 🛠️ Setup & Run

Run this in your terminal:

```
# 1. Create virtual environment (first time only)
python -m venv ghost_env

# 2. Activate environment
.\ghost_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run ghostbusters_app.py

```

## 📁 What's Inside

```
├── models/                          # Trained model files (created when you run)
├── ghostbusters_app.py              # The Streamlit dashboard
├── run_ghostbusters.py              # Auto-setup script
├── requirements.txt                 # Packages you need
└── dating_app_behavior_dataset.csv  # Your data

```

## 🔧 What We Used

* **Models:** Random Forest, Neural Network, Logistic Regression, Decision Tree, SVM
* **Explainability:** SHAP values
* **Balancing:** SMOTE
* **Visuals:** Streamlit + Plotly

## 📚 Course Info

* **WIA1006/WID3006 Machine Learning** - Semester 2, 2025/2026
