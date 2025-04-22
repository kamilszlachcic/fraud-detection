from pathlib import Path

# === Project structure ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # => /app/src

# === Directories ===
MODELS_DIR = PROJECT_ROOT / "src" / "models"
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = PROJECT_ROOT / "data/raw"
DATA_PROCESSING_DIR = PROJECT_ROOT / "src" / "data_processing"

# === Ensure directories exist ===
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === Specific files ===
XGB_MODEL_PATH = MODELS_DIR / "XGB_model.pkl"
PREDICTION_LOG_PATH = LOG_DIR / "predictions.log.jsonl"
