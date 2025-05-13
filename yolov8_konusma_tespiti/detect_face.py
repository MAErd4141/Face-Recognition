import face_recognition
import cv2
import os
import sys
import time
import numpy as np

# Yol ayarı
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agiz_analiz.agiz_tespit import agiz_acikligi
from duygu_analizi.analyze_emotion import predict_emotion

# ============ Bilinen Yüzleri Yükle ============
KNOWN_DIR = "../known_faces"
known_face_encodings = []
known_face_names = []

for file_name in os.listdir(KNOWN_DIR):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        path = os.path.join(KNOWN_DIR, file_name)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(file_name)[0].capitalize()
            known_face_names.append(name)

# ============ Takip Değişkenleri ============
konusma_durumlari = {}
konusma_baslangic = {}
toplam_konusma_suresi = {}
son_konusma_zamani = {}

# ============ Kamera Aç ============
print("📷 Kamera başlatılıyor...")
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("❌ Kamera açılmadı!")
    time.sleep(2)
    exit()
else:
    print("✅ Kamera başarıyla açıldı.")

# ============ Ana Döngü ============
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Görüntü alınamadı.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Bilinmeyen"
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Duygu tahmini
        face_crop = frame[top:bottom, left:right]
        emotion = predict_emotion(face_crop)

        # Ağız açıklığını hesapla
        try:
            top_lip = face_landmarks.get("top_lip")
            bottom_lip = face_landmarks.get("bottom_lip")

            if top_lip and bottom_lip:
                aciklik = agiz_acikligi(np.array(top_lip), np.array(bottom_lip))
                konusuyor = aciklik > 7.0
                print(f"[AĞIZ] {name} → Açıklık: {aciklik:.2f}")
            else:
                raise ValueError("Ağız noktaları eksik")
        except Exception as e:
            print(f"[HATA] Ağız açıklığı hesaplanamadı: {e}")
            continue

        # Kişi ilk kez görülüyorsa
        if name not in konusma_durumlari:
            konusma_durumlari[name] = False
            konusma_baslangic[name] = None
            toplam_konusma_suresi[name] = 0
            son_konusma_zamani[name] = 0

        # Konuşma başladı mı?
        if konusuyor:
            if not konusma_durumlari[name]:
                konusma_baslangic[name] = time.time()
                konusma_durumlari[name] = True
                print(f"[BAŞLADI] {name} konuşmaya başladı")
            son_konusma_zamani[name] = time.time()

        # Konuşma bitti mi?
        elif konusma_durumlari[name]:
            if time.time() - son_konusma_zamani[name] > 1.0:
                if konusma_baslangic[name] is not None:
                    sure = time.time() - konusma_baslangic[name]
                    toplam_konusma_suresi[name] += sure
                    print(f"[BİTTİ] {name} konuşmayı bitirdi. +{sure:.2f} sn")
                konusma_durumlari[name] = False
                konusma_baslangic[name] = None

        # Görüntü üzerine yaz
        sure_sn = int(toplam_konusma_suresi.get(name, 0))
        sure_str = time.strftime("%M:%S", time.gmtime(sure_sn))
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} | {emotion} | {sure_str}", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Yüz Tanıma + Duygu + Ağızla Konuşma Süresi", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# ============ Program Kapanırken: Süreyi Kurtar ============
for name in konusma_durumlari:
    if konusma_durumlari[name] and konusma_baslangic[name] is not None:
        sure = time.time() - konusma_baslangic[name]
        toplam_konusma_suresi[name] += sure
        print(f"[BİTTİ - KAPANIŞ] {name} konuşmayı bitirdi. +{sure:.2f} sn")

video_capture.release()
cv2.destroyAllWindows()
print("🧯 Sistem kapatıldı.")
