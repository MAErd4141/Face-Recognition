# track.py
import numpy as np

class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = []

    def update(self, detections):
        updated_tracks = []

        for det in detections:
            x1, y1, x2, y2, conf = det

            matched = False
            for track in self.tracks:
                tx1, ty1, tx2, ty2, track_id = track
                # Merkezler arası mesafe ile eşleştirme
                dist = np.linalg.norm(np.array([(x1+x2)/2, (y1+y2)/2]) - np.array([(tx1+tx2)/2, (ty1+ty2)/2]))
                if dist < 50:  # Eşik değeri
                    updated_tracks.append([x1, y1, x2, y2, track_id])
                    matched = True
                    break

            if not matched:
                updated_tracks.append([x1, y1, x2, y2, self.next_id])
                self.next_id += 1

        self.tracks = updated_tracks
        return np.array(self.tracks)
