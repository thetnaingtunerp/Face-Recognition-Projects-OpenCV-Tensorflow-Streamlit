import io
import time
from datetime import date, timedelta

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from config import FACES_DIR, MATCH_THRESHOLD, ATTENDANCE_COOLDOWN_SECONDS
from database import (
    init_db, add_employee, get_employees, add_embedding, get_all_embeddings,
    record_check_in, record_check_out, get_attendance, attendance_summary,
    set_employee_active
)
from face_engine import FaceEngine

st.set_page_config(page_title="Face Recognition Attendance", page_icon="🧑‍💼", layout="wide")
init_db()
FACES_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_engine():
    return FaceEngine()

def image_to_bgr(uploaded):
    image = Image.open(uploaded).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def save_face_sample(employee_id, face_rgb, embedding):
    employee_dir = FACES_DIR / employee_id
    employee_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(employee_dir.glob("*.jpg"))) + 1
    Image.fromarray(face_rgb).save(employee_dir / f"face_{count:03d}.jpg", quality=95)
    add_embedding(employee_id, embedding)

def page_dashboard():
    st.title("📊 Attendance Dashboard")
    summary = attendance_summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Employees", summary["total"])
    c2.metric("Present Today", summary["present"])
    c3.metric("Absent Today", summary["absent"])
    rows = get_attendance(date.today(), date.today())
    st.subheader("Today's Attendance")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Attendance is recorded once per employee per calendar day. Use the Attendance Records page to enter a check-out.")

def page_employees(engine):
    st.title("👥 Employee Registration")
    with st.form("employee_form", clear_on_submit=True):
        employee_id = st.text_input("Employee ID *")
        full_name = st.text_input("Full name *")
        department = st.text_input("Department")
        email = st.text_input("Email")
        submitted = st.form_submit_button("Create Employee", type="primary")
    if submitted:
        if not employee_id.strip() or not full_name.strip():
            st.error("Employee ID and full name are required.")
        else:
            try:
                add_employee(employee_id, full_name, department, email)
                st.success("Employee created. Capture face samples below.")
            except Exception as exc:
                st.error(f"Could not create employee: {exc}")

    employees = get_employees(active_only=False)
    if employees:
        st.subheader("Employee Directory")
        df = pd.DataFrame(employees)
        st.dataframe(df[["employee_id", "full_name", "department", "email", "active", "created_at"]],
                     use_container_width=True, hide_index=True)
        st.subheader("Capture Face Samples")
        active = [e for e in employees if e["active"]]
        if active:
            selected = st.selectbox("Select employee", active, format_func=lambda e: f'{e["employee_id"]} — {e["full_name"]}')
            capture = st.camera_input("Take a clear, front-facing face photo", key="registration_camera")
            if capture:
                bgr = image_to_bgr(capture)
                boxes = engine.detect(bgr)
                if len(boxes) != 1:
                    st.warning(f"Please capture exactly one face. Detected: {len(boxes)}")
                else:
                    face_rgb = engine.crop_face_rgb(bgr, boxes[0])
                    st.image(face_rgb, caption="Detected face crop", width=220)
                    if st.button("Save this face sample", type="primary"):
                        embedding = engine.embedding_from_rgb(face_rgb)
                        save_face_sample(selected["employee_id"], face_rgb, embedding)
                        st.success("Face sample and FaceNet embedding saved. Add 3–5 samples with different lighting/angles.")
        inactive = [e for e in employees if not e["active"]]
        if inactive:
            st.subheader("Inactive Employees")
            for e in inactive:
                st.write(f'{e["employee_id"]} — {e["full_name"]}')

class AttendanceVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.engine = load_engine()
        self.known = get_all_embeddings()
        self.last_seen = {}
        self.last_refresh = 0
        self.last_message = "Waiting for camera…"

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        now = time.time()
        if now - self.last_refresh > 10:
            self.known = get_all_embeddings()
            self.last_refresh = now
        boxes = self.engine.detect(image)
        for box in boxes:
            x, y, w, h = box
            try:
                face_rgb = self.engine.crop_face_rgb(image, box)
                match, distance = self.engine.recognize(face_rgb, self.known, MATCH_THRESHOLD)
                if match:
                    employee_id = match["employee_id"]
                    label = f'{match["full_name"]} ({distance:.2f})'
                    if now - self.last_seen.get(employee_id, 0) > ATTENDANCE_COOLDOWN_SECONDS:
                        success, msg = record_check_in(employee_id)
                        self.last_seen[employee_id] = now
                        self.last_message = f'{employee_id}: {msg}'
                    color = (0, 180, 0)
                else:
                    label = "Unknown"
                    color = (0, 0, 220)
            except Exception as exc:
                label, color = f"Error: {str(exc)[:30]}", (0, 165, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label, (x, max(25, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, self.last_message[:90], (10, image.shape[0]-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        return av.VideoFrame.from_ndarray(image, format="bgr24")

def page_live_recognition():
    st.title("📷 Live Face Recognition & Check-in")
    st.warning("Use this feature only with informed employee consent. This prototype does not include liveness detection; a printed photo or screen replay may spoof it.")
    known = get_all_embeddings()
    if not known:
        st.info("No face embeddings are registered yet. Register employees and capture face samples first.")
        return
    st.caption(f"Recognition threshold: Euclidean distance ≤ {MATCH_THRESHOLD:.2f}. Calibrate this value with your own test images before operational use.")
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_streamer(
        key="attendance-camera",
        video_processor_factory=AttendanceVideoProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.caption("A successful first match creates today's check-in. Repeated detections do not create duplicate daily rows.")

def page_records():
    st.title("🗓️ Attendance Records")
    c1, c2 = st.columns(2)
    start = c1.date_input("From", value=date.today() - timedelta(days=30))
    end = c2.date_input("To", value=date.today())
    if start > end:
        st.error("From date must be before or equal to To date.")
        return
    rows = get_attendance(start, end)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if not df.empty:
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"attendance_{start}_{end}.csv", mime="text/csv")
    st.subheader("Record Check-out")
    employees = get_employees()
    if employees:
        selected = st.selectbox("Employee", employees,
                                format_func=lambda e: f'{e["employee_id"]} — {e["full_name"]}')
        if st.button("Record Check-out", type="primary"):
            if record_check_out(selected["employee_id"]):
                st.success("Check-out recorded.")
            else:
                st.warning("No open check-in found for this employee today.")
    else:
        st.info("No active employees.")

def page_settings():
    st.title("⚙️ Employee Status & Settings")
    st.write(f"**Face match threshold:** {MATCH_THRESHOLD}")
    st.write("Lower threshold = stricter matching. Tune with genuine and impostor test samples.")
    st.subheader("Activate / Deactivate Employees")
    for emp in get_employees(active_only=False):
        c1, c2, c3 = st.columns([2, 4, 2])
        c1.write(emp["employee_id"])
        c2.write(emp["full_name"])
        new_state = c3.toggle("Active", value=bool(emp["active"]), key=f'active_{emp["employee_id"]}')
        if new_state != bool(emp["active"]):
            set_employee_active(emp["employee_id"], new_state)
            st.rerun()

def main():
    st.sidebar.title("Face Attendance")
    page = st.sidebar.radio("Navigation", [
        "Dashboard", "Employee Registration", "Live Recognition", "Attendance Records", "Settings"
    ])
    try:
        engine = load_engine() if page in ("Employee Registration", "Live Recognition") else None
        if page == "Dashboard": page_dashboard()
        elif page == "Employee Registration": page_employees(engine)
        elif page == "Live Recognition": page_live_recognition()
        elif page == "Attendance Records": page_records()
        else: page_settings()
    except Exception as exc:
        st.error(f"Application error: {exc}")
        st.exception(exc)
        st.info("Check the installation steps and terminal output. First FaceNet model initialization may take time and download pretrained weights.")

if __name__ == "__main__":
    main()
