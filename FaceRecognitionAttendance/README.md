# Face Recognition Attendance System

A local prototype for employee attendance using Python 3.12, TensorFlow/Keras FaceNet embeddings, OpenCV Haar-cascade face detection, Streamlit, WebRTC webcam streaming and SQLite.

## Features

- Employee master records (ID, name, department, email)
- Face sample capture from the browser camera
- FaceNet embedding generation and SQLite storage
- Live webcam recognition in Streamlit
- One check-in per employee per calendar day
- Manual check-out
- Attendance date filtering and CSV export
- Employee activation/deactivation

## Architecture

Browser webcam -> Streamlit WebRTC -> OpenCV face detection/crop -> TensorFlow FaceNet embedding -> Euclidean-distance matching -> SQLite attendance.

Haar cascade is used only for face detection. FaceNet produces embeddings for identity comparison.

## Requirements

- Python 3.12 (64-bit)
- Windows 10/11, Linux, or macOS
- Webcam and a modern browser
- Internet for initial package/pretrained model download
- For smoother inference, a supported GPU is optional; CPU works but may be slower.

## Installation (Windows PowerShell)

```powershell
cd FaceRecognitionAttendance
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Installation (macOS/Linux)

```bash
cd FaceRecognitionAttendance
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501). Allow camera permission in the browser.

## First-time setup

1. Open **Employee Registration**.
2. Create an employee with a unique ID, for example `EMP001`.
3. Use **Take a clear, front-facing face photo** and save the sample.
4. Repeat to save 3–5 samples per employee. Use consistent lighting and a few natural angles.
5. Open **Live Recognition**, click Start, and stand in front of the camera.
6. The system records the first recognized check-in for the current date.
7. Use **Attendance Records** to review/export records and record check-out.

FaceNet's pretrained weights may be downloaded/cached on first initialization. Allow time and internet access for this first run.

## Database schema

### employees
- `employee_id` TEXT PRIMARY KEY
- `full_name` TEXT NOT NULL
- `department`, `email` TEXT
- `active` INTEGER
- `created_at` TEXT

### face_embeddings
- `id` INTEGER PRIMARY KEY
- `employee_id` TEXT foreign key -> employees
- `embedding` BLOB (float32 vector)
- `created_at` TEXT

### attendance
- `id` INTEGER PRIMARY KEY
- `employee_id` TEXT foreign key -> employees
- `attendance_date` TEXT (YYYY-MM-DD)
- `check_in`, `check_out` TEXT (ISO datetime)
- `recognition_method` TEXT
- UNIQUE(`employee_id`, `attendance_date`)

SQLite file is created automatically at `data/attendance.db`.

## Recognition threshold

The starter threshold is Euclidean distance `0.75` for L2-normalized FaceNet embeddings. This is a starting point, not a universal setting. Camera quality, lighting, face alignment, model version and environment affect distances. Test genuine and non-matching identities, then tune `MATCH_THRESHOLD` in `config.py`. A lower value is stricter and may increase false rejections; a higher value may increase false matches.

## Testing checklist

1. Create two test employees with unique IDs.
2. Capture at least 3 face samples for each employee.
3. Verify `data/attendance.db` exists and each employee has embeddings.
4. Start live recognition and test each enrolled person separately.
5. Confirm the overlay shows the correct name and a green box.
6. Test an unregistered person; it should show `Unknown`.
7. Verify a single attendance row is created for the recognized employee for today.
8. Keep the same person in view; verify duplicate daily attendance is not created.
9. Test check-out from Attendance Records.
10. Export CSV and verify employee ID, name, date, check-in and check-out.
11. Test low light, side angles, glasses and different distances.
12. Re-test the threshold after changing the camera or environment.

## Important limitations and production hardening

This is a runnable starter prototype, not a production-grade biometric security product.

- No liveness / anti-spoofing is implemented. A photo or screen replay may be accepted.
- Haar cascade can miss faces or return imperfect crops. For higher accuracy, replace it with an aligned-face detector such as MTCNN or RetinaFace.
- FaceNet threshold must be calibrated locally; do not assume the default is accurate for every workforce.
- This implementation stores biometric embeddings and face sample images locally without encryption or role-based access control.
- Obtain informed consent, document retention/deletion rules, restrict filesystem/database access, encrypt backups, and provide a manual attendance correction process.
- Do not use recognition as the sole basis for pay, discipline, or other consequential decisions.
- Streamlit's local app should be deployed behind HTTPS and authentication if accessed over a network. Browser camera access generally requires localhost or a secure HTTPS origin.
- For production, add admin authentication, audit logs, employee data deletion, consent tracking, liveness checks, model evaluation, backup/restore and concurrency testing.

## Troubleshooting

**TensorFlow / package installation fails**
- Confirm 64-bit Python 3.12: `python --version` and `python -c "import struct; print(struct.calcsize('P')*8)"`.
- Upgrade pip and install in a fresh virtual environment.
- TensorFlow wheel availability depends on OS, architecture and Python version.

**Camera does not start**
- Allow camera permission in browser.
- Close other apps using the webcam.
- Use `localhost` or HTTPS; camera access may be blocked on insecure remote HTTP pages.
- For remote deployments, configure WebRTC STUN/TURN connectivity.

**FaceNet first launch is slow**
- The pretrained model weights may need to download on first use. Check internet access and retry.

**Wrong identity / unknown**
- Capture more samples in even lighting.
- Keep face frontal and unobstructed.
- Tune `MATCH_THRESHOLD` with a validation set. Do not increase it blindly.

## Project structure

```text
FaceRecognitionAttendance/
├── app.py
├── config.py
├── database.py
├── face_engine.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/                 # created automatically
    ├── attendance.db
    └── employee_faces/
```
