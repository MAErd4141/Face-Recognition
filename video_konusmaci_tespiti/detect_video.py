import cv2
import face_recognition
import dlib
import numpy as np
import os
from collections import defaultdict
import sys

# agiz_analiz klasörünü modül yoluna ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'agiz_analiz')))
from agiz_tespit import agiz_acikligi


# Sabitler
KNOWN_FACES_DIR = r"C:\YPY\known_faces"
LANDMARK_PATH = r"C:\YPY\shape_predictor_68_face_landmarks.dat"
VIDEO_PATH = r"C:\YPY\video_konusmaci_tespiti\ornek_video.mp4"
MOUTH_OPEN_THRESHOLD = 0.5

# Yüz tanıma setup
known_face_encodings = []
known_face_names = []
talking_times = defaultdict(float)

# Modeller
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARK_PATH)

# Bilinen yüzleri yükle
def load_known_faces():
    for filename in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(image)
        if encs:
            known_face_encodings.append(encs[0])
            known_face_names.append(os.path.splitext(filename)[0].capitalize())

load_known_faces()

# Video aç
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locs = face_recognition.face_locations(rgb_small)
    encs = face_recognition.face_encodings(rgb_small, locs)

    for enc, loc in zip(encs, locs):
        top, right, bottom, left = [v * 2 for v in loc]
        name = "Bilinmeyen"
        matches = face_recognition.compare_faces(known_face_encodings, enc)
        face_distances = face_recognition.face_distance(known_face_encodings, enc)
        if len(face_distances) > 0:
            best_match = np.argmin(face_distances)
            if matches[best_match]:
                name = known_face_names[best_match]

        rect = dlib.rectangle(left, top, right, bottom)
        shape = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rect)
        landmarks = [(p.x, p.y) for p in shape.parts()]

        oran = agiz_acikligi(landmarks)
        if oran > MOUTH_OPEN_THRESHOLD:
            talking_times[name] += 1 / fps

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n🧾 Konuşma Süreleri:")
for kisi, sure in talking_times.items():
    print(f"{kisi}: {sure:.2f} saniye")
