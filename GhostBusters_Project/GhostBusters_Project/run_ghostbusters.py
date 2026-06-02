#!/usr/bin/env python3
"""
GhostBusters - Complete Auto-Runner
This script will automatically:
1. Create or activate a virtual environment called ghost_env
2. Install required packages if missing
3. Generate a synthetic dataset if needed
4. Train five models and save the Random Forest
5. Create the Streamlit app file
6. Launch the Streamlit app on localhost:8501
"""

import os
import sys
import subprocess
import traceback
import webbrowser
import venv
import importlib.util
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / 'dating_app_behavior_dataset.csv'
MODELS_DIR = SCRIPT_DIR / 'models'
BEST_MODEL_PATH = MODELS_DIR / 'best_model.pkl'
PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.pkl'
COMPARISON_PATH = MODELS_DIR / 'model_comparison.csv'
APP_PATH = SCRIPT_DIR / 'ghostbusters_app.py'
VENV_DIR = SCRIPT_DIR / 'ghost_env'
VENV_PYTHON = VENV_DIR / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

REQUIRED_PACKAGES = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scikit-learn': 'sklearn',
    'streamlit': 'streamlit',
    'plotly': 'plotly',
    'imbalanced-learn': 'imblearn',
    'joblib': 'joblib',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'shap': 'shap'
}


def check_python_version():
    if sys.version_info < (3, 8):
        raise RuntimeError('Python 3.8 or newer is required. Please install a supported Python version.')
    print(f'✅ Python {sys.version_info.major}.{sys.version_info.minor} detected')


def is_venv_active():
    try:
        return Path(sys.executable).resolve() == VENV_PYTHON.resolve()
    except Exception:
        return False


def create_virtualenv():
    if VENV_DIR.exists() and VENV_PYTHON.exists():
        print(f'✅ Virtual environment already exists at {VENV_DIR}')
        return

    print('🛠️  Creating virtual environment ghost_env...')
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(VENV_DIR)
    if not VENV_PYTHON.exists():
        raise RuntimeError(f'Failed to create virtual environment executable at {VENV_PYTHON}')
    print('✅ Virtual environment created')


def restart_in_virtualenv():
    if is_venv_active():
        return

    if not VENV_PYTHON.exists():
        create_virtualenv()

    print('🔁 Restarting script inside ghost_env...')
    env = os.environ.copy()
    env['GHOSTBUSTERS_REEXEC'] = '1'
    result = subprocess.run([str(VENV_PYTHON), str(Path(__file__).resolve())] + sys.argv[1:], env=env)
    sys.exit(result.returncode)


def install_packages():
    print('\n📦 Checking required packages...')
    missing_packages = []

    for package, module_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package)
        else:
            print(f'  ✓ {package} already installed')

    if not missing_packages:
        print('✅ All required packages are already installed')
        return

    print('⬇ Installing missing packages: ' + ', '.join(missing_packages))
    for package in missing_packages:
        command = [sys.executable, '-m', 'pip', 'install', package]
        print(f'    Installing {package}...')
        subprocess.check_call(command)

    print('✅ Package installation complete')


def create_dataset():
    if DATA_PATH.exists():
        print(f'📊 Dataset already exists at {DATA_PATH.name}')
        return

    print('🔨 Generating synthetic dataset in the existing schema...')
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 10000
    categories = {
        'gender': ['Female', 'Male', 'Non-binary', 'Genderfluid', 'Transgender', 'Prefer Not to Say'],
        'sexual_orientation': ['Straight', 'Gay', 'Lesbian', 'Bisexual', 'Pansexual', 'Asexual', 'Demisexual', 'Queer'],
        'location_type': ['Urban', 'Suburban', 'Rural', 'Metro', 'Small Town', 'Remote Area'],
        'income_bracket': ['Very Low', 'Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High', 'Very High'],
        'education_level': ['High School', 'Bachelor’s', 'Master’s', 'PhD', 'Postdoc', 'MBA', 'Associate’s', 'Diploma', 'No Formal Education'],
        'app_usage_time_label': ['Very Low', 'Low', 'Moderate', 'High', 'Extreme User', 'Barely', 'Addicted'],
        'swipe_right_label': ['Balanced', 'Choosy', 'Optimistic', 'Swipe Maniac'],
        'swipe_time_of_day': ['Morning', 'Afternoon', 'Evening', 'Late Night', 'After Midnight', 'Early Morning'],
        'match_outcome': ['Ghosted', 'Chat Ignored', 'No Action', 'One-sided Like', 'Mutual Match', 'Date Happened', 'One-sided Like']
    }

    df = pd.DataFrame({
        'gender': rng.choice(categories['gender'], size=n),
        'sexual_orientation': rng.choice(categories['sexual_orientation'], size=n),
        'location_type': rng.choice(categories['location_type'], size=n),
        'income_bracket': rng.choice(categories['income_bracket'], size=n),
        'education_level': rng.choice(categories['education_level'], size=n),
        'interest_tags': [', '.join(rng.choice(['Traveling', 'Fitness', 'Music', 'Cooking', 'Gaming', 'Movies', 'Reading', 'Art'], size=rng.integers(1, 5), replace=False)) for _ in range(n)],
        'app_usage_time_min': rng.integers(0, 301, size=n),
        'app_usage_time_label': rng.choice(categories['app_usage_time_label'], size=n),
        'swipe_right_ratio': rng.random(size=n),
        'swipe_right_label': rng.choice(categories['swipe_right_label'], size=n),
        'likes_received': rng.integers(0, 501, size=n),
        'mutual_matches': rng.integers(0, 101, size=n),
        'profile_pics_count': rng.integers(1, 11, size=n),
        'bio_length': rng.integers(0, 501, size=n),
        'message_sent_count': rng.integers(0, 201, size=n),
        'emoji_usage_rate': rng.random(size=n),
        'last_active_hour': rng.integers(0, 24, size=n),
        'swipe_time_of_day': rng.choice(categories['swipe_time_of_day'], size=n),
        'match_outcome': rng.choice(categories['match_outcome'], size=n)
    })
    df.to_csv(DATA_PATH, index=False)
    print(f'✅ Created {DATA_PATH.name} with {len(df):,} records')


def train_models():
    if BEST_MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() and COMPARISON_PATH.exists():
        print(f'📦 Model artifacts already exist in {MODELS_DIR}. Skipping training.')
        return

    print('\n🤖 Training models...')
    import pandas as pd
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from imblearn.over_sampling import SMOTE
    import joblib

    df = pd.read_csv(DATA_PATH)
    required_columns = {
        'app_usage_time_min', 'swipe_right_ratio', 'likes_received',
        'mutual_matches', 'profile_pics_count', 'bio_length',
        'message_sent_count', 'emoji_usage_rate', 'last_active_hour',
        'gender', 'sexual_orientation', 'education_level', 'income_bracket',
        'location_type', 'interest_tags', 'app_usage_time_label',
        'swipe_right_label', 'swipe_time_of_day', 'match_outcome'
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f'Dataset is missing required columns: {sorted(missing)}')

    if 'interest_count' not in df.columns:
        df['interest_count'] = df['interest_tags'].fillna('').apply(lambda value: len([tag for tag in str(value).split(',') if tag.strip()]))

    if df['swipe_right_ratio'].max() <= 1.0:
        df['swipe_right_ratio'] = df['swipe_right_ratio'] * 100
    if df['emoji_usage_rate'].max() <= 1.0:
        df['emoji_usage_rate'] = df['emoji_usage_rate'] * 100

    ghost_labels = ['ghosted', 'chat ignored', 'no action', 'one-sided like', 'one sided like', 'no response', 'unmatched']
    df['is_ghosted'] = df['match_outcome'].fillna('').astype(str).str.lower().apply(
        lambda text: int(any(label in text for label in ghost_labels))
    )

    feature_columns = [
        'app_usage_time_min', 'swipe_right_ratio', 'likes_received',
        'mutual_matches', 'profile_pics_count', 'bio_length',
        'message_sent_count', 'emoji_usage_rate', 'last_active_hour',
        'interest_count', 'gender', 'sexual_orientation', 'education_level',
        'income_bracket', 'location_type', 'app_usage_time_label',
        'swipe_right_label', 'swipe_time_of_day'
    ]
    numeric_features = [
        'app_usage_time_min', 'swipe_right_ratio', 'likes_received',
        'mutual_matches', 'profile_pics_count', 'bio_length',
        'message_sent_count', 'emoji_usage_rate', 'last_active_hour',
        'interest_count'
    ]
    categorical_features = [
        'gender', 'sexual_orientation', 'education_level', 'income_bracket',
        'location_type', 'app_usage_time_label', 'swipe_right_label',
        'swipe_time_of_day'
    ]

    X = df[feature_columns]
    y = df['is_ghosted'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])

    print('  🧹 Preprocessing data...')
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = feature_columns

    print('  ⚖️ Balancing classes with SMOTE...')
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_transformed, y_train)

    candidate_models = {
        'Logistic Regression': LogisticRegression(
            solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced', random_state=42
        ),
        'Support Vector Machine': SVC(
            probability=True, class_weight='balanced', random_state=42, gamma='scale', max_iter=1000
        ),
        'Neural Network (MLP)': MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True
        )
    }

    results = []
    for name, model in candidate_models.items():
        print(f'  🚧 Training {name}...')
        model.fit(X_train_balanced, y_train_balanced)
        y_pred = model.predict(X_test_transformed)
        y_prob = model.predict_proba(X_test_transformed)[:, 1]
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'AUC_ROC': roc_auc_score(y_test, y_prob)
        })
        print(f'    ✓ {name} complete')

    print('  🔍 Training and tuning Random Forest...')
    rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 12, None],
        'min_samples_split': [2, 5]
    }
    grid = GridSearchCV(rf, param_grid, scoring='f1', cv=3, n_jobs=-1, verbose=0)
    grid.fit(X_train_balanced, y_train_balanced)
    best_rf = grid.best_estimator_
    y_pred = best_rf.predict(X_test_transformed)
    y_prob = best_rf.predict_proba(X_test_transformed)[:, 1]
    results.append({
        'Model': 'Random Forest (Best)',
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC_ROC': roc_auc_score(y_test, y_prob)
    })
    print(f'    ✓ Random Forest best params: {grid.best_params_}')

    results_df = pd.DataFrame(results).sort_values(by='F1', ascending=False).reset_index(drop=True)
    MODELS_DIR.mkdir(exist_ok=True)
    print(f'  💾 Saving model artifacts to {MODELS_DIR}')
    joblib.dump(best_rf, BEST_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    results_df.to_csv(COMPARISON_PATH, index=False)

    print('\n✅ Training complete')
    print(results_df.to_string(index=False))


def create_streamlit_app():
    print('\n📝 Writing Streamlit app file...')
    content = """import streamlit as st
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
"""
    APP_PATH.write_text(content, encoding='utf-8')
    print(f'✅ Streamlit app file written to {APP_PATH.name}')


def launch_app():
    print('\n🚀 Launching Streamlit app...')
    url = 'http://localhost:8501'
    print(f'🔗 Opening {url}')
    try:
        webbrowser.open(url)
    except Exception:
        print('⚠️ Could not open browser automatically. Use the URL above.')

    subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(APP_PATH), '--server.port', '8501'], check=True)


def main():
    try:
        print('=' * 58)
        print('👻 GhostBusters Setup & Launch')
        print('=' * 58)
        check_python_version()
        create_virtualenv()
        restart_in_virtualenv()
        install_packages()
        create_dataset()
        train_models()
        create_streamlit_app()
        launch_app()
    except Exception as exc:
        print('\n❌ GhostBusters failed:')
        print(str(exc))
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
