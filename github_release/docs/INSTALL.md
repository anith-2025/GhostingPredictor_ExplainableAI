# GhostBusters — Quick install & run (packaged release)

Follow the same instructions in the project root `INSTALL.md`. This packaged copy is intended
for a tidy GitHub release layout where `backend/` contains scripts and `frontend/` contains the app.

Run from the repository root:

```bash
cd github_release/backend
python -m venv ghost_env
source ghost_env/bin/activate    # or .\ghost_env\Scripts\activate.bat on Windows
pip install -r ../../requirements.txt
python run_ghostbusters.py
```
