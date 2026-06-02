# GhostBusters — Quick install & run

Minimal steps for a new user to get the project running locally.

Prerequisites
- Python 3.8+ installed and on PATH
- Git (optional) to clone the repo

Quickstart (recommended)

Windows (PowerShell):

```powershell
python -m venv ghost_env
.\ghost_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_ghostbusters.py
```

macOS / Linux (bash):

```bash
python3 -m venv ghost_env
source ghost_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_ghostbusters.py
```

Notes
- `run_ghostbusters.py` will also attempt to create a `ghost_env` virtualenv and install missing packages automatically. If you prefer manual control or encounter permission issues, use the steps above first.
- If you see ModuleNotFoundError for packages like `numpy` or `shap`, run the above `pip install -r requirements.txt` inside the activated `ghost_env`.

Troubleshooting
- If Streamlit doesn't start, ensure the venv is active and `streamlit` appears in `pip list`.
- On Windows, enable script execution in PowerShell if activation fails: run PowerShell as Administrator and `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Optional helper scripts
- `install_requirements.bat` — quick Windows batch installer
- `install_requirements.sh` — quick POSIX installer

If you'd like, I can also add a GitHub Actions workflow to validate the install steps on each push.
