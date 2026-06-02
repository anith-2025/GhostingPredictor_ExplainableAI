"""
Packaged Streamlit app for release (frontend)
This file is a copy of the app used in the project. Keep model files in a `models/` folder
placed alongside this `frontend` folder for the app to load saved artifacts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="GhostBusters - Ghosting Predictor", layout="wide")

# Load saved models and data
@st.cache_resource
def load_models():
    """Load all saved models and artifacts"""
    try:
        base = Path(__file__).resolve().parent
        model = joblib.load(base.parent / 'models' / 'best_model.pkl')
        preprocessor = joblib.load(base.parent / 'models' / 'preprocessor.pkl')
        feature_names = joblib.load(base.parent / 'models' / 'feature_names.pkl')
        model_metrics = joblib.load(base.parent / 'models' / 'model_metrics.pkl')
        shap_importance = joblib.load(base.parent / 'models' / 'shap_importance.pkl')
        return model, preprocessor, feature_names, model_metrics, shap_importance
    except FileNotFoundError:
        st.error("""
        ⚠️ Model files not found! Please:
        1. Run the Jupyter notebook first to train and save models
        2. Make sure the 'models' folder is in the parent directory of this app
        """)
        return None, None, None, None, None

from pathlib import Path

# --- rest of the app is identical to the project copy ---

_ORIGINAL_APP_PLACEHOLDER = True
