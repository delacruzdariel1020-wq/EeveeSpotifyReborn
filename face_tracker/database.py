"""
Face database: stores and retrieves faces by their feature encodings.

Each face is saved as:
  - name: human-readable label
  - encoding: 128-dim feature vector (from face_recognition)
  - landmarks: dict of facial landmark positions (eyes, nose, mouth, chin…)
  - metadata: timestamp, source image path, etc.
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import Optional


DB_PATH = os.path.join(os.path.dirname(__file__), "faces_db.json")
TOLERANCE = 0.55  # lower = stricter match


class FaceDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.faces: list[dict] = []
        self._load()

    # ------------------------------------------------------------------ I/O

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                raw = json.load(f)
            self.faces = raw.get("faces", [])
            # Convert encoding lists back to numpy arrays
            for face in self.faces:
                face["encoding"] = np.array(face["encoding"])
        else:
            self.faces = []

    def save(self):
        serializable = []
        for face in self.faces:
            entry = dict(face)
            entry["encoding"] = face["encoding"].tolist()
            serializable.append(entry)
        with open(self.db_path, "w") as f:
            json.dump({"faces": serializable}, f, indent=2)

    # ----------------------------------------------------------------- CRUD

    def add_face(
        self,
        name: str,
        encoding: np.ndarray,
        landmarks: Optional[dict] = None,
        source: str = "",
    ) -> dict:
        """Save a new face. Returns the stored entry."""
        entry = {
            "name": name,
            "encoding": encoding,
            "landmarks": landmarks or {},
            "source": source,
            "added_at": datetime.utcnow().isoformat(),
        }
        self.faces.append(entry)
        self.save()
        return entry

    def remove_face(self, name: str) -> int:
        """Remove all entries with this name. Returns count removed."""
        before = len(self.faces)
        self.faces = [f for f in self.faces if f["name"] != name]
        self.save()
        return before - len(self.faces)

    def list_names(self) -> list[str]:
        return sorted({f["name"] for f in self.faces})

    # --------------------------------------------------------------- SEARCH

    def find_match(
        self, encoding: np.ndarray, tolerance: float = TOLERANCE
    ) -> Optional[dict]:
        """
        Return the closest stored face within tolerance, or None.
        Uses Euclidean distance on 128-dim encoding vectors.
        """
        if not self.faces:
            return None

        stored = np.array([f["encoding"] for f in self.faces])
        distances = np.linalg.norm(stored - encoding, axis=1)
        best_idx = int(np.argmin(distances))

        if distances[best_idx] <= tolerance:
            return {
                "face": self.faces[best_idx],
                "distance": float(distances[best_idx]),
            }
        return None

    def find_all_matches(
        self, encoding: np.ndarray, tolerance: float = TOLERANCE
    ) -> list[dict]:
        """Return all stored faces within tolerance, sorted by distance."""
        if not self.faces:
            return []

        stored = np.array([f["encoding"] for f in self.faces])
        distances = np.linalg.norm(stored - encoding, axis=1)

        matches = [
            {"face": self.faces[i], "distance": float(distances[i])}
            for i in range(len(self.faces))
            if distances[i] <= tolerance
        ]
        matches.sort(key=lambda m: m["distance"])
        return matches

    # ------------------------------------------------------------ LANDMARKS

    @staticmethod
    def summarize_landmarks(landmarks: dict) -> dict:
        """
        Convert raw landmark point lists into summary metrics:
        inter-eye distance, eye centres, nose tip, mouth centre, etc.
        These are the 'rasgos' (features) that characterize a face.
        """
        def centre(points: list) -> list:
            arr = np.array(points)
            return arr.mean(axis=0).tolist()

        summary = {}

        if "left_eye" in landmarks:
            summary["left_eye_centre"] = centre(landmarks["left_eye"])
        if "right_eye" in landmarks:
            summary["right_eye_centre"] = centre(landmarks["right_eye"])

        if "left_eye_centre" in summary and "right_eye_centre" in summary:
            le = np.array(summary["left_eye_centre"])
            re = np.array(summary["right_eye_centre"])
            summary["inter_eye_distance"] = float(np.linalg.norm(le - re))
            summary["eye_midpoint"] = ((le + re) / 2).tolist()

        if "nose_tip" in landmarks:
            summary["nose_tip"] = centre(landmarks["nose_tip"])
        if "nose_bridge" in landmarks:
            summary["nose_bridge_top"] = landmarks["nose_bridge"][0]

        if "top_lip" in landmarks:
            summary["mouth_centre"] = centre(landmarks["top_lip"])
        if "chin" in landmarks:
            summary["chin_tip"] = landmarks["chin"][len(landmarks["chin"]) // 2]

        return summary
