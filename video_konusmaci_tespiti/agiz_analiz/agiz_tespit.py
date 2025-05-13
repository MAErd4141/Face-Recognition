# agiz_tespit.py
import numpy as np

def agiz_acikligi(landmarks):
    try:
        top_lip = np.array(landmarks[62])
        bottom_lip = np.array(landmarks[66])
        face_top = np.array(landmarks[27])
        face_bottom = np.array(landmarks[8])

        mouth_open_dist = np.linalg.norm(top_lip - bottom_lip)
        face_height = np.linalg.norm(face_top - face_bottom)

        return mouth_open_dist / face_height
    except:
        return 0
