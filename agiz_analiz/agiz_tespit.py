import numpy as np

def agiz_acikligi(top_lip, bottom_lip):
    top_lip_center = np.mean(top_lip, axis=0)
    bottom_lip_center = np.mean(bottom_lip, axis=0)
    return np.linalg.norm(top_lip_center - bottom_lip_center)
