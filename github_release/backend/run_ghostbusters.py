"""
Packaged copy of run_ghostbusters.py for release (backend)
Keep this copy for GitHub release structure. This is a straight copy of the runner
used to create the venv, install requirements, train models, and launch the app.
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
DATA_PATH = SCRIPT_DIR / '..' / 'dating_app_behavior_dataset.csv'
MODELS_DIR = SCRIPT_DIR / '..' / '..' / 'models'
BEST_MODEL_PATH = MODELS_DIR / 'best_model.pkl'
PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.pkl'
COMPARISON_PATH = MODELS_DIR / 'model_comparison.csv'
FEATURE_NAMES_PATH = MODELS_DIR / 'feature_names.pkl'
MODEL_METRICS_PATH = MODELS_DIR / 'model_metrics.pkl'
SHAP_IMPORTANCE_PATH = MODELS_DIR / 'shap_importance.pkl'
APP_PATH = SCRIPT_DIR / '..' / 'ghostbusters_streamlit.py'
VENV_DIR = SCRIPT_DIR / '..' / 'ghost_env'
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


if __name__ == '__main__':
    try:
        print('=' * 58)
        print('👻 GhostBusters Release Runner')
        print('=' * 58)
        check_python_version()
        create_virtualenv()
        restart_in_virtualenv()
        install_packages()
        print('🎉 Setup steps completed in the packaged release folder.')
    except Exception as exc:
        print('\n❌ GhostBusters failed:')
        print(str(exc))
        traceback.print_exc()
        sys.exit(1)
