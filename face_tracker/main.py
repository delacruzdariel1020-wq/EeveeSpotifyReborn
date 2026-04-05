import cv2
import face_recognition
import numpy as np
import time
from face_db import FaceDatabase

# --- Config ---
RECOGNITION_THRESHOLD = 0.45   # Mas estricto = mas preciso
PROCESS_EVERY_N_FRAMES = 2     # Cada 2 frames para mayor fluidez
SCAN_SAMPLES = 20              # Muestras a capturar al guardar una cara
SCAN_SECONDS = 3.0             # Duracion del escaneo en segundos
NUM_JITTERS = 3                # Repeticiones internas para mayor precision (mas lento pero mejor)
DB_PATH = "faces_db.sqlite"

COLOR_KNOWN   = (0, 220, 0)
COLOR_UNKNOWN = (0, 100, 255)
COLOR_TEXT    = (255, 255, 255)
COLOR_SCAN    = (0, 220, 220)


def lerp(a, b, t):
    return int(a + (b - a) * t)


def smooth_bbox(prev, curr, alpha=0.4):
    if prev is None:
        return curr
    return tuple(lerp(p, c, alpha) for p, c in zip(prev, curr))


def draw_bbox(frame, top, right, bottom, left, label, color, similarity=None):
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    corner = 10
    for (x, y) in [(left, top), (right, top), (left, bottom), (right, bottom)]:
        dx = 1 if x == left else -1
        dy = 1 if y == top else -1
        cv2.line(frame, (x, y), (x + dx * corner, y), color, 3)
        cv2.line(frame, (x, y), (x, y + dy * corner), color, 3)

    label_text = label
    if similarity is not None:
        label_text += f"  {similarity:.0%}"
    label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.rectangle(frame, (left, top - 24), (left + label_size[0] + 8, top), color, cv2.FILLED)
    cv2.putText(frame, label_text, (left + 4, top - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_TEXT, 1)


def draw_scan_overlay(frame, progress: float, samples: int):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    radius = 90
    angle = int(360 * progress)
    cv2.ellipse(frame, (cx, cy), (radius, radius), -90, 0, angle, COLOR_SCAN, 4)
    cv2.circle(frame, (cx, cy), radius - 8, (0, 0, 0), cv2.FILLED)
    text = "Escaneando..."
    sub  = f"{samples}/{SCAN_SAMPLES} muestras"
    ts, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ss, _ = cv2.getTextSize(sub,  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, text, (cx - ts[0]//2, cy - 8),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_SCAN, 2)
    cv2.putText(frame, sub,  (cx - ss[0]//2, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)


def draw_hud(frame, total_faces: int, detected: int):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 44), (w, h), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    status = "  [S] Guardar cara  |  [L] Listar  |  [Q] Salir"
    cv2.putText(frame, status, (10, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT, 1)
    info = f"DB: {total_faces} caras  |  En camara: {detected}"
    cv2.putText(frame, info, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)


def capture_face_encoding(cap, loc_small):
    encodings = []
    start = time.time()
    while len(encodings) < SCAN_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb)
        if locs:
            encs = face_recognition.face_encodings(rgb, locs, num_jitters=NUM_JITTERS)
            if encs:
                encodings.append(encs[0])

        progress = len(encodings) / SCAN_SAMPLES
        display  = frame.copy()
        draw_scan_overlay(display, progress, len(encodings))
        cv2.imshow("Face Tracker", display)
        cv2.waitKey(1)

        if time.time() - start > SCAN_SECONDS * 2:
            break

    if not encodings:
        return None
    return np.mean(encodings, axis=0)


def main():
    db = FaceDatabase(DB_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: no se pudo abrir la camara.")
        return

    print("Face Tracker iniciado.")
    print("  S -> escanear y guardar cara (3 seg, 20 muestras)")
    print("  L -> listar caras guardadas")
    print("  Q -> salir\n")

    known_faces  = db.load_all_faces()
    frame_count  = 0
    prev_bboxes  = {}

    face_locations    = []
    face_labels       = []
    face_colors       = []
    face_similarities = []
    face_ids          = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small   = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            locs      = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, locs, num_jitters=NUM_JITTERS)

            new_locations     = []
            new_labels        = []
            new_colors        = []
            new_similarities  = []
            new_ids           = []

            known_encodings = [f["encoding"] for f in known_faces]

            for idx, (loc, enc) in enumerate(zip(locs, encodings)):
                top, right, bottom, left = [v * 2 for v in loc]
                raw_bbox = (top, right, bottom, left)
                smooth   = smooth_bbox(prev_bboxes.get(idx), raw_bbox, alpha=0.35)
                prev_bboxes[idx] = smooth
                new_locations.append(smooth)

                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, enc)
                    best_idx  = int(np.argmin(distances))
                    best_dist = distances[best_idx]

                    if best_dist <= RECOGNITION_THRESHOLD:
                        match = known_faces[best_idx]
                        new_labels.append(match["name"])
                        new_colors.append(COLOR_KNOWN)
                        new_similarities.append(1.0 - best_dist)
                        new_ids.append(match["id"])
                        db.log_sighting(match["id"], loc)
                    else:
                        new_labels.append("Desconocido")
                        new_colors.append(COLOR_UNKNOWN)
                        new_similarities.append(None)
                        new_ids.append(None)
                else:
                    new_labels.append("Desconocido")
                    new_colors.append(COLOR_UNKNOWN)
                    new_similarities.append(None)
                    new_ids.append(None)

            prev_bboxes = {i: v for i, v in prev_bboxes.items() if i < len(locs)}

            face_locations    = new_locations
            face_labels       = new_labels
            face_colors       = new_colors
            face_similarities = new_similarities
            face_ids          = new_ids

        for i, bbox in enumerate(face_locations):
            top, right, bottom, left = bbox
            label = face_labels[i]       if i < len(face_labels)       else "?"
            color = face_colors[i]       if i < len(face_colors)       else COLOR_UNKNOWN
            sim   = face_similarities[i] if i < len(face_similarities) else None
            draw_bbox(frame, top, right, bottom, left, label, color, sim)

        draw_hud(frame, db.get_face_count(), len(face_locations))
        cv2.imshow("Face Tracker", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            unknown_indices = [i for i, lbl in enumerate(face_labels) if lbl == "Desconocido"]
            if not unknown_indices:
                print("No hay caras desconocidas en pantalla.")
                continue

            idx = unknown_indices[0]
            loc = face_locations[idx]
            small_loc = tuple(v // 2 for v in loc)

            print(f"\nEscaneando cara... ({SCAN_SAMPLES} muestras)")
            avg_encoding = capture_face_encoding(cap, small_loc)

            if avg_encoding is None:
                print("  No se pudo escanear. Intentalo de nuevo.")
                continue

            print(f"Nombre para la nueva cara (DB tiene {db.get_face_count()} caras):")
            name = input("  Nombre: ").strip()
            if not name:
                name = f"Persona_{db.get_face_count() + 1}"

            face_id = db.save_face(name, avg_encoding)
            print(f"  Guardado: {name} (ID {face_id}, {SCAN_SAMPLES} muestras promediadas)\n")
            known_faces = db.load_all_faces()

        elif key == ord('l'):
            print("\n--- Caras guardadas ---")
            for f in db.list_faces():
                print(f"  [{f['id']}] {f['name']}  |  {f['created_at'][:10]}  |  {f['sightings']} avistamientos")
            print()

    cap.release()
    cv2.destroyAllWindows()
    print("Face Tracker cerrado.")


if __name__ == "__main__":
    main()
