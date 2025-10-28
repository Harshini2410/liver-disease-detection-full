Liver Disease Detection - Backend

This folder contains a minimal Flask backend that integrates a PyTorch model for liver disease detection and provides API endpoints compatible with the frontend.

Quick start (Windows PowerShell):

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the API server:

```powershell
python app.py
```

The server will listen on http://localhost:5000. The frontend is configured with a proxy so requests to `/api/*` are forwarded to this server when running `npm start` in the frontend.

Notes:
- The repo includes a model file named `final_complete_hope.pt` in the original backend. Place that file into this folder if you want the model to load. If it's missing, the backend will run in demo/fallback mode.
- Results are persisted to `./results/` as JSON files and uploaded images are saved to `./uploads/`.
