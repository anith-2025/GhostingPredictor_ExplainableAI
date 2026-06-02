# GhostBusters — Packaged Release

This folder is a tidy packaged layout intended for a GitHub release. It separates frontend and backend components
so new contributors can quickly see what belongs where.

Structure
- `backend/` — scripts to create venv, install dependencies and run training (`run_ghostbusters.py`)
- `frontend/` — Streamlit app (`ghostbusters_streamlit.py`)
- `docs/` — `INSTALL.md` with quickstart instructions for this packaged layout
- `scripts/` — helpers to install requirements on Windows/macOS/Linux

Notes
- The packaged files are copies for clarity; the working project lives at the repository root.
- To run the packaged release locally, follow `docs/INSTALL.md`.
