# 👻 GhostBusters

Ghosting predictor for dating apps. Uses 5 ML models + SHAP explainability.

## 🚀 Quick Start (One Command)

Run this in your terminal:

```bash
python run_ghostbusters.py
```

That's it. The script sets everything up, trains the models, and opens the app in your browser.

## 🛠️ Manual Setup

If you want to do things step by step:

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train models**
- Open `WIA1006_GhostingPredictor_ExplainableAI_v2_fixed.ipynb`
- Run all cells

**3. Launch app**
```bash
streamlit run ghostbusters_app.py
```

## 📁 What's Inside

```
├── models/                    # Trained model files (created when you run)
├── ghostbusters_app.py        # The Streamlit dashboard
├── run_ghostbusters.py        # Auto-setup script
├── requirements.txt           # Packages you need
└── dating_app_behavior_dataset.csv  # Your data
```

## 🔧 What We Used

- **Models**: Random Forest, Neural Network, Logistic Regression, Decision Tree, SVM
- **Explainability**: SHAP values
- **Balancing**: SMOTE
- **Visuals**: Streamlit + Plotly

## 📚 Course Info

WIA1006/WID3006 Machine Learning - Semester 2, 2025/2026
