import threading
import cv2
import numpy as np
from keras_facenet import FaceNet
from config import FACE_SIZE, DETECTION_SCALE, DETECTION_NEIGHBORS, MIN_FACE_SIZE

class FaceEngine:
    """OpenCV Haar face detector + pretrained TensorFlow/Keras FaceNet embeddings."""
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
        self.embedder = FaceNet()
        self._model_lock = threading.Lock()

    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detectMultiScale(
            gray, scaleFactor=DETECTION_SCALE,
            minNeighbors=DETECTION_NEIGHBORS,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
        )
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in boxes]

    def crop_face_rgb(self, frame_bgr, box, margin=15):
        x, y, w, h = box
        height, width = frame_bgr.shape[:2]
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(width, x+w+margin), min(height, y+h+margin)
        face = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        if face.size == 0:
            raise ValueError("Empty face crop")
        face = cv2.resize(face, FACE_SIZE, interpolation=cv2.INTER_AREA)
        return face.astype(np.uint8)

    def embedding_from_rgb(self, face_rgb):
        batch = np.expand_dims(face_rgb.astype(np.float32), axis=0)
        with self._model_lock:
            embeddings = self.embedder.embeddings(batch)
        vector = np.asarray(embeddings[0], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Face embedding has zero norm")
        return vector / norm

    def recognize(self, face_rgb, known_embeddings, threshold=0.75):
        if not known_embeddings:
            return None, None
        query = self.embedding_from_rgb(face_rgb)
        best_id, best_name, best_distance = None, None, float('inf')
        for item in known_embeddings:
            ref = item['embedding']
            ref_norm = np.linalg.norm(ref)
            if ref_norm == 0:
                continue
            distance = float(np.linalg.norm(query - (ref / ref_norm)))
            if distance < best_distance:
                best_id, best_name, best_distance = item['employee_id'], item['full_name'], distance
        if best_distance <= threshold:
            return {"employee_id": best_id, "full_name": best_name}, best_distance
        return None, best_distance
