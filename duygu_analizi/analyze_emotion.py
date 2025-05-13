from deepface import DeepFace
from fer import FER
import cv2

# FER dedektörünü başlat
detector_fer = FER(mtcnn=True)

# Duygu etiketleri: İngilizce → Türkçe
duygu_etiketleri = {
    "angry": "kızgın",
    "happy": "mutlu",
    "sad": "üzgün",
    "neutral": "nötr",
    "disgust": "nötr",
    "fear": "nötr",
    "surprise": "mutlu"
}

def predict_emotion(img):
    try:
        # DeepFace analizi
        deep_result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)
        deep_emotion = deep_result[0].get("dominant_emotion", "neutral")

        # FER analizi
        fer_result = detector_fer.detect_emotions(img)
        if fer_result:
            fer_emotion = max(fer_result[0]["emotions"], key=fer_result[0]["emotions"].get)
        else:
            fer_emotion = "neutral"

        # Karar mantığı (A seçeneği gibi):
        if "happy" in [deep_emotion, fer_emotion, "surprise"]:
            secilen = "happy"
        elif {"angry", "sad"} & {deep_emotion, fer_emotion}:
            secilen = deep_emotion if deep_emotion in ["angry", "sad"] else fer_emotion
        else:
            secilen = "neutral"

        return duygu_etiketleri.get(secilen, "nötr")

    except Exception as e:
        print("[Duygu Hatası]", e)
        return "nötr"
