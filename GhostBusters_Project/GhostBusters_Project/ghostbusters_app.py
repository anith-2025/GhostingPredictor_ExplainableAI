import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / 'models' / 'best_model.pkl'
PREPROCESSOR_PATH = SCRIPT_DIR / 'models' / 'preprocessor.pkl'
COMPARISON_PATH = SCRIPT_DIR / 'models' / 'model_comparison.csv'

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        comparison = pd.read_csv(COMPARISON_PATH)
    except FileNotFoundError:
        st.error('⚠️ Required artifacts are missing. Please run python run_ghostbusters.py first.')
        st.stop()
    except Exception as exc:
        st.error(f'⚠️ Failed to load artifacts: {exc}')
        st.stop()
    return model, preprocessor, comparison

model, preprocessor, comparison = load_artifacts()

FEATURE_INFO = {
    'app_usage_time_min': {'label': 'App Usage Time (min/day)', 'min': 0, 'max': 360, 'value': 90},
    'swipe_right_ratio': {'label': 'Swipe Right Ratio (%)', 'min': 0, 'max': 100, 'value': 45},
    'likes_received': {'label': 'Likes Received', 'min': 0, 'max': 500, 'value': 60},
    'mutual_matches': {'label': 'Mutual Matches', 'min': 0, 'max': 100, 'value': 10},
    'profile_pics_count': {'label': 'Profile Pictures', 'min': 1, 'max': 10, 'value': 4},
    'bio_length': {'label': 'Bio Length (characters)', 'min': 0, 'max': 500, 'value': 150},
    'message_sent_count': {'label': 'Messages Sent (per day)', 'min': 0, 'max': 200, 'value': 40},
    'emoji_usage_rate': {'label': 'Emoji Usage Rate (%)', 'min': 0, 'max': 100, 'value': 40},
    'last_active_hour': {'label': 'Last Active Hour', 'min': 0, 'max': 23, 'value': 18},
    'interest_count': {'label': 'Interest Count', 'min': 1, 'max': 12, 'value': 5}
}

CATEGORICAL_OPTIONS = {
    'gender': ['Female', 'Male', 'Non-binary', 'Genderfluid', 'Transgender', 'Prefer Not to Say'],
    'sexual_orientation': ['Straight', 'Gay', 'Lesbian', 'Bisexual', 'Pansexual', 'Asexual', 'Demisexual', 'Queer'],
    'education_level': ['High School', 'Bachelor’s', 'Master’s', 'PhD', 'Postdoc', 'MBA', 'Associate’s', 'Diploma', 'No Formal Education'],
    'income_bracket': ['Very Low', 'Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High', 'Very High'],
    'location_type': ['Urban', 'Suburban', 'Rural', 'Metro', 'Small Town', 'Remote Area'],
    'app_usage_time_label': ['Very Low', 'Low', 'Moderate', 'High', 'Extreme User', 'Barely', 'Addicted'],
    'swipe_right_label': ['Balanced', 'Choosy', 'Optimistic', 'Swipe Maniac'],
    'swipe_time_of_day': ['Morning', 'Afternoon', 'Evening', 'Late Night', 'After Midnight', 'Early Morning']
}

st.set_page_config(page_title='GhostBusters Ghosting Predictor', layout='wide')
st.title('👻 GhostBusters')
st.markdown('AI-powered ghosting risk predictor with explainable model insights.')

section = st.sidebar.radio('Navigation', ['🎯 Live Predictor', '📊 Model Comparison', '📖 About'])

@st.cache_resource
def build_shap_explainer(model):
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return shap.Explainer(model)

if section == '🎯 Live Predictor':
    st.subheader('Live Ghosting Risk Predictor')
    with st.form('prediction_form'):
        cols = st.columns(2)
        user_input = {}
        for idx, (key, cfg) in enumerate(FEATURE_INFO.items()):
            column = cols[idx % 2]
            user_input[key] = column.slider(cfg['label'], cfg['min'], cfg['max'], cfg['value'])

        user_input['gender'] = cols[0].selectbox('Gender', CATEGORICAL_OPTIONS['gender'])
        user_input['sexual_orientation'] = cols[0].selectbox('Sexual Orientation', CATEGORICAL_OPTIONS['sexual_orientation'])
        user_input['education_level'] = cols[1].selectbox('Education Level', CATEGORICAL_OPTIONS['education_level'])
        user_input['income_bracket'] = cols[1].selectbox('Income Bracket', CATEGORICAL_OPTIONS['income_bracket'])
        user_input['location_type'] = cols[0].selectbox('Location Type', CATEGORICAL_OPTIONS['location_type'])
        user_input['app_usage_time_label'] = cols[1].selectbox('Usage Style', CATEGORICAL_OPTIONS['app_usage_time_label'])
        user_input['swipe_right_label'] = cols[0].selectbox('Swipe Style', CATEGORICAL_OPTIONS['swipe_right_label'])
        user_input['swipe_time_of_day'] = cols[1].selectbox('Swipe Time of Day', CATEGORICAL_OPTIONS['swipe_time_of_day'])

        submitted = st.form_submit_button('🔮 Predict Ghosting Risk')

    if submitted:
        input_df = pd.DataFrame([user_input])
        try:
            processed = preprocessor.transform(input_df)
        except Exception as exc:
            st.error(f'⚠️ Failed to process input values: {exc}')
            st.stop()

        probability = float(model.predict_proba(processed)[:, 1][0])
        risk_label = 'High Ghosting Risk' if probability >= 0.5 else 'Low Ghosting Risk'
        st.metric('Prediction', risk_label, f'{probability:.1%}')

        gauge = go.Figure(go.Indicator(
            mode='gauge+number',
            value=probability * 100,
            title={'text': 'Ghosting Risk Score'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#b22222' if probability >= 0.5 else '#228b22'},
                'steps': [
                    {'range': [0, 33], 'color': 'lightgreen'},
                    {'range': [33, 66], 'color': 'gold'},
                    {'range': [66, 100], 'color': 'lightcoral'}
                ]
            }
        ))
        gauge.update_layout(height=320)
        st.plotly_chart(gauge, use_container_width=True)

        explainer = build_shap_explainer(model)
        try:
            shap_values = explainer(processed)
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0]
            else:
                values = shap_values[0]
        except Exception:
            shap_values = explainer.shap_values(processed)
            values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        feature_names = [
            name.replace('numeric__', '').replace('categorical__', '')
            for name in preprocessor.get_feature_names_out()
        ]
        shap_df = pd.DataFrame({'feature': feature_names, 'shap_value': values})
        shap_df['abs_value'] = shap_df['shap_value'].abs()
        shap_df = shap_df.sort_values('abs_value', ascending=False).head(10)

        st.subheader('SHAP Feature Importance')
        shap_fig = px.bar(
            shap_df.sort_values('abs_value'),
            x='abs_value', y='feature', orientation='h',
            color='shap_value', color_continuous_scale='RdBu',
            labels={'abs_value': 'Impact', 'feature': 'Feature', 'shap_value': 'SHAP Value'},
            title='Top 10 features influencing this prediction'
        )
        st.plotly_chart(shap_fig, use_container_width=True)

        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            st.subheader('Random Forest Feature Importance')
            importance_fig = px.bar(
                importance_df.sort_values('importance'),
                x='importance', y='feature', orientation='h',
                labels={'importance': 'Importance', 'feature': 'Feature'},
                title='Top 10 Random Forest features'
            )
            st.plotly_chart(importance_fig, use_container_width=True)

elif section == '📊 Model Comparison':
    st.subheader('Model Comparison and Metrics')
    st.write('Evaluation metrics for all trained models:')
    st.dataframe(comparison, use_container_width=True)
    st.markdown('''
        - **F1** balances precision and recall.
        - **AUC_ROC** evaluates prediction ranking.
        - The Random Forest model is saved as the main predictor.
    ''')

else:
    st.subheader('About GhostBusters')
    st.markdown('''
        ### Project Overview
        GhostBusters predicts dating app ghosting behavior using a behavior-first model.

        Ghosting is when a conversation partner suddenly stops replying.

        The pipeline includes synthetic dataset creation, preprocessing, SMOTE balancing, model training, and explainable prediction.

        Run `python run_ghostbusters.py` to regenerate data, train, and launch the app.
    ''')
