from ultralytics import YOLO
import cv2
import numpy as np
from track import SimpleTracker
import time
import sys
import os
from collections import deque, Counter

# Özel duygu analiz modülü
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from duygu_analizi.analyze_emotion import predict_emotion

# Model ve takip sistemi
model = YOLO("yolov8n.pt")
tracker = SimpleTracker()

# Sözlükler
id_giris_sureleri = {}
id_toplam_sure = {}
id_son_duygular = {}
id_son_analiz_zamani = {}
id_duygu_gecmisi = {}

# Kamera başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    detections = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        detections.append([x1, y1, x2, y2, score])

    tracked = tracker.update(np.array(detections))

    for d in tracked:
        x1, y1, x2, y2, track_id = map(int, d)

        if track_id not in id_giris_sureleri:
            id_giris_sureleri[track_id] = time.time()
            id_toplam_sure[track_id] = 0
            id_son_duygular[track_id] = "..."
            id_son_analiz_zamani[track_id] = 0
            id_duygu_gecmisi[track_id] = deque(maxlen=5)

        goruntude_kalma_suresi = time.time() - id_giris_sureleri[track_id]
        id_toplam_sure[track_id] = goruntude_kalma_suresi

        # Her 2 saniyede bir duygu analizi
        if time.time() - id_son_analiz_zamani[track_id] > 2:
            padding = 20
            y1_c = max(0, y1 - padding)
            y2_c = min(frame.shape[0], y2 + padding)
            x1_c = max(0, x1 - padding)
            x2_c = min(frame.shape[1], x2 + padding)
            face_crop = frame[y1_c:y2_c, x1_c:x2_c]

            if face_crop.size > 0:
                emotion = predict_emotion(face_crop)
                id_duygu_gecmisi[track_id].append(emotion)

                en_sik_duygu = Counter(id_duygu_gecmisi[track_id]).most_common(1)[0][0]
                id_son_duygular[track_id] = en_sik_duygu
                id_son_analiz_zamani[track_id] = time.time()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track_id} | {int(goruntude_kalma_suresi)} sn | {id_son_duygular[track_id]}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Yüz Takibi + Süre + Duygu (TR - temiz)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
