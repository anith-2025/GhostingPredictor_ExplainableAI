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
FEATURE_NAMES_PATH = MODELS_DIR / 'feature_names.pkl'
MODEL_METRICS_PATH = MODELS_DIR / 'model_metrics.pkl'
SHAP_IMPORTANCE_PATH = MODELS_DIR / 'shap_importance.pkl'
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
    if (BEST_MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() and COMPARISON_PATH.exists()
            and FEATURE_NAMES_PATH.exists() and MODEL_METRICS_PATH.exists() and SHAP_IMPORTANCE_PATH.exists()):
        print(f'📦 Model artifacts already exist in {MODELS_DIR}. Skipping training.')
        return

    print('\n🤖 Training models...')
    import pandas as pd
    import numpy as np
    import joblib
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
    import shap

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
        df['interest_count'] = df['interest_tags'].fillna('').apply(
            lambda value: len([tag for tag in str(value).split(',') if tag.strip()])
        )

    if df['swipe_right_ratio'].max() <= 1.0:
        df['swipe_right_ratio'] = df['swipe_right_ratio'] * 100
    if df['emoji_usage_rate'].max() <= 1.0:
        df['emoji_usage_rate'] = df['emoji_usage_rate'] * 100

    df['is_ghosted'] = (
        df['match_outcome'].astype(str).str.strip() == 'Ghosted'
    ).astype(int)

    feature_columns = [
        'app_usage_time_min', 'swipe_right_ratio', 'likes_received',
        'mutual_matches', 'profile_pics_count', 'bio_length',
        'message_sent_count', 'emoji_usage_rate', 'last_active_hour',
        'interest_count', 'gender', 'sexual_orientation', 'location_type',
        'income_bracket', 'education_level', 'app_usage_time_label',
        'swipe_right_label', 'swipe_time_of_day'
    ]

    numeric_features = [
        'app_usage_time_min', 'swipe_right_ratio', 'likes_received',
        'mutual_matches', 'profile_pics_count', 'bio_length',
        'message_sent_count', 'emoji_usage_rate', 'last_active_hour',
        'interest_count'
    ]

    categorical_features = [
        'gender', 'sexual_orientation', 'location_type', 'income_bracket',
        'education_level', 'app_usage_time_label', 'swipe_right_label',
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
        feature_names = numeric_features.copy()
        onehot = preprocessor.named_transformers_['categorical'].named_steps['onehot']
        feature_names.extend(onehot.get_feature_names_out(categorical_features))

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
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    joblib.dump(results_df.set_index('Model').to_dict(orient='index'), MODEL_METRICS_PATH)
    results_df.to_csv(COMPARISON_PATH, index=False)

    print('  📦 Computing SHAP importance...')
    try:
        explainer = shap.TreeExplainer(best_rf)
        shap_values = explainer(X_test_transformed[:500])
    except Exception:
        explainer = shap.Explainer(best_rf, X_train_balanced)
        shap_values = explainer(X_test_transformed[:500])

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    if hasattr(shap_values, 'values'):
        shap_array = shap_values.values
    else:
        shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    mean_abs = np.mean(np.abs(shap_array), axis=0)
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    joblib.dump(shap_df, SHAP_IMPORTANCE_PATH)

    print('\n✅ Training complete')
    print(results_df.to_string(index=False))


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
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
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
        launch_app()
    except Exception as exc:
        print('\n❌ GhostBusters failed:')
        print(str(exc))
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()