from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "employee_faces"
DB_PATH = DATA_DIR / "attendance.db"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODEL_CACHE_DIR = BASE_DIR / "models"

FACE_SIZE = (160, 160)       # FaceNet input size
DETECTION_SCALE = 1.1
DETECTION_NEIGHBORS = 5
MATCH_THRESHOLD = 0.75       # Calibrate using your camera and employees
MIN_FACE_SIZE = 60
ATTENDANCE_COOLDOWN_SECONDS = 60
