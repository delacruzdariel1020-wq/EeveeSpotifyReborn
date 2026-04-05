"""
Core face tracker.

Modes:
  live   – webcam loop; press 's' to save the highlighted face, 'q' to quit.
  image  – process a single image file.
  video  – process a video file (no interactive save).
"""

import cv2
import face_recognition
import numpy as np
from typing import Optional

from database import FaceDatabase


# BGR colours for display
COLOUR_KNOWN   = (0, 200, 0)
COLOUR_UNKNOWN = (0, 60, 255)
COLOUR_LM      = (0, 220, 255)
FONT           = cv2.FONT_HERSHEY_SIMPLEX


def _rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _draw_face(
    frame: np.ndarray,
    top: int, right: int, bottom: int, left: int,
    label: str,
    colour: tuple,
    distance: Optional[float] = None,
):
    cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)
    if distance is not None:
        label = f"{label} ({distance:.2f})"
    cv2.rectangle(frame, (left, bottom - 24), (right, bottom), colour, cv2.FILLED)
    cv2.putText(frame, label, (left + 4, bottom - 6), FONT, 0.55, (255, 255, 255), 1)


def _draw_landmarks(frame: np.ndarray, landmarks: dict):
    for feature, points in landmarks.items():
        pts = np.array(points, dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], isClosed=False, color=COLOUR_LM, thickness=1)
        else:
            cv2.circle(frame, tuple(pts[0]), 2, COLOUR_LM, -1)


def process_frame(
    frame: np.ndarray,
    db: FaceDatabase,
    show_landmarks: bool = True,
    scale: float = 0.5,
) -> tuple[np.ndarray, list[dict]]:
    """
    Detect all faces in frame, match against db, draw annotations.
    Returns (annotated_frame, list of face_info dicts).

    Each face_info contains:
      location, encoding, landmarks, landmark_summary, match (or None)
    """
    small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small = _rgb(small)

    locations  = face_recognition.face_locations(rgb_small, model="hog")
    encodings  = face_recognition.face_encodings(rgb_small, locations)
    landmarks_list = face_recognition.face_landmarks(rgb_small, locations)

    face_infos = []
    for loc, enc, lm in zip(locations, encodings, landmarks_list):
        top, right, bottom, left = [int(v / scale) for v in loc]

        # Scale landmarks back to original resolution
        scaled_lm = {
            feat: [(int(x / scale), int(y / scale)) for x, y in pts]
            for feat, pts in lm.items()
        }

        summary = FaceDatabase.summarize_landmarks(scaled_lm)
        match   = db.find_match(enc)

        if match:
            label  = match["face"]["name"]
            colour = COLOUR_KNOWN
            dist   = match["distance"]
        else:
            label  = "Desconocido"
            colour = COLOUR_UNKNOWN
            dist   = None

        _draw_face(frame, top, right, bottom, left, label, colour, dist)
        if show_landmarks:
            _draw_landmarks(frame, scaled_lm)

        face_infos.append({
            "location": (top, right, bottom, left),
            "encoding": enc,
            "landmarks": scaled_lm,
            "landmark_summary": summary,
            "match": match,
        })

    return frame, face_infos


# ------------------------------------------------------------------ LIVE MODE

def run_live(db: FaceDatabase, show_landmarks: bool = True, camera_index: int = 0):
    """
    Webcam live loop.
      s  – save the largest face with a name you type in the terminal
      l  – toggle landmark display
      q  – quit
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    show_lm = show_landmarks
    print("\n[FaceTracker] Controles: s=guardar cara | l=landmarks | q=salir\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, face_infos = process_frame(frame, db, show_lm)

        status = f"Caras: {len(face_infos)}  |  DB: {len(db.faces)} registros  |  [s] guardar  [l] landmarks  [q] salir"
        cv2.putText(annotated, status, (8, 20), FONT, 0.5, (220, 220, 220), 1)

        cv2.imshow("EeveeFaceTracker", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("l"):
            show_lm = not show_lm

        elif key == ord("s") and face_infos:
            # Pick largest face by bounding-box area
            def area(fi):
                t, r, b, l = fi["location"]
                return (b - t) * (r - l)

            target = max(face_infos, key=area)
            cv2.destroyAllWindows()
            name = input("Nombre para esta cara: ").strip()
            if name:
                db.add_face(
                    name=name,
                    encoding=target["encoding"],
                    landmarks=target["landmark_summary"],
                )
                print(f"  Guardada: '{name}'  (distancia encoding ≈ 0.00 – es nueva)")
            else:
                print("  Cancelado (nombre vacío).")
            cv2.imshow("EeveeFaceTracker", annotated)

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------- IMAGE MODE

def run_image(
    path: str,
    db: FaceDatabase,
    show_landmarks: bool = True,
    save_name: Optional[str] = None,
    output_path: Optional[str] = None,
):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"No se pudo abrir: {path}")

    annotated, face_infos = process_frame(frame, db, show_landmarks, scale=1.0)

    print(f"\nDetectadas {len(face_infos)} cara(s) en '{path}':")
    for i, fi in enumerate(face_infos):
        match = fi["match"]
        name  = match["face"]["name"] if match else "Desconocido"
        dist  = f"{match['distance']:.3f}" if match else "—"
        t, r, b, l = fi["location"]
        print(f"  [{i}] {name}  dist={dist}  bbox=({l},{t})→({r},{b})")
        lms = fi["landmark_summary"]
        for feat, val in lms.items():
            print(f"       {feat}: {val}")

    if save_name and face_infos:
        # Save encoding of the largest face
        def area(fi):
            t, r, b, l = fi["location"]
            return (b - t) * (r - l)
        target = max(face_infos, key=area)
        db.add_face(
            name=save_name,
            encoding=target["encoding"],
            landmarks=target["landmark_summary"],
            source=path,
        )
        print(f"\nGuardada cara más grande como '{save_name}'.")

    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"Imagen anotada guardada en: {output_path}")
    else:
        cv2.imshow("EeveeFaceTracker", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
