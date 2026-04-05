import cv2
import face_recognition
import numpy as np
from face_db import FaceDatabase

# --- Config ---
RECOGNITION_THRESHOLD = 0.50   # Distancia maxima para considerar match (menor = mas estricto)
PROCESS_EVERY_N_FRAMES = 3     # Procesar 1 de cada N frames para mejor rendimiento
DB_PATH = "faces_db.sqlite"

# Colores BGR
COLOR_KNOWN   = (0, 220, 0)    # Verde para caras conocidas
COLOR_UNKNOWN = (0, 100, 255)  # Naranja para desconocidas
COLOR_TEXT    = (255, 255, 255)
COLOR_SAVING  = (255, 255, 0)  # Cian al guardar


def draw_bbox(frame, top, right, bottom, left, label, color, similarity=None):
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    label_text = label
    if similarity is not None:
        label_text += f"  {similarity:.0%}"
    label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.rectangle(frame, (left, top - 22), (left + label_size[0] + 6, top), color, cv2.FILLED)
    cv2.putText(frame, label_text, (left + 3, top - 6),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, COLOR_TEXT, 1)


def draw_hud(frame, total_faces: int, detected: int, saving_mode: bool):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    status = "  [S] Guardar cara  |  [Q] Salir  |  [L] Listar caras"
    if saving_mode:
        status = "  MODO GUARDAR: escribe el nombre en la terminal..."
    cv2.putText(frame, status, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

    info = f"Caras en DB: {total_faces}  |  Detectadas: {detected}"
    cv2.putText(frame, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)


def main():
    db = FaceDatabase(DB_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: no se pudo abrir la camara.")
        return

    print("Face Tracker iniciado.")
    print("  S -> guardar cara detectada")
    print("  L -> listar caras en DB")
    print("  Q -> salir\n")

    known_faces = db.load_all_faces()
    frame_count = 0

    face_locations = []
    face_labels = []
    face_colors = []
    face_similarities = []
    face_ids = []

    saving_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            locs = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, locs)

            face_locations = []
            face_labels = []
            face_colors = []
            face_similarities = []
            face_ids = []

            known_encodings = [f["encoding"] for f in known_faces]

            for loc, enc in zip(locs, encodings):
                top, right, bottom, left = [v * 2 for v in loc]
                face_locations.append((top, right, bottom, left))

                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, enc)
                    best_idx = int(np.argmin(distances))
                    best_dist = distances[best_idx]

                    if best_dist <= RECOGNITION_THRESHOLD:
                        match = known_faces[best_idx]
                        face_labels.append(match["name"])
                        face_colors.append(COLOR_KNOWN)
                        face_similarities.append(1.0 - best_dist)
                        face_ids.append(match["id"])
                        db.log_sighting(match["id"], loc)
                    else:
                        face_labels.append("Desconocido")
                        face_colors.append(COLOR_UNKNOWN)
                        face_similarities.append(None)
                        face_ids.append(None)
                else:
                    face_labels.append("Desconocido")
                    face_colors.append(COLOR_UNKNOWN)
                    face_similarities.append(None)
                    face_ids.append(None)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            label = face_labels[i] if i < len(face_labels) else "?"
            color = face_colors[i] if i < len(face_colors) else COLOR_UNKNOWN
            sim   = face_similarities[i] if i < len(face_similarities) else None
            draw_bbox(frame, top, right, bottom, left, label, color, sim)

        draw_hud(frame, db.get_face_count(), len(face_locations), saving_mode)
        cv2.imshow("Face Tracker", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            unknown_indices = [i for i, lbl in enumerate(face_labels) if lbl == "Desconocido"]
            if not unknown_indices:
                print("No hay caras desconocidas en pantalla para guardar.")
                continue

            idx = unknown_indices[0]
            top, right, bottom, left = face_locations[idx]
            small_loc = (top // 2, right // 2, bottom // 2, left // 2)
            enc_list = face_recognition.face_encodings(rgb_small, [small_loc])
            if not enc_list:
                print("No se pudo extraer encoding.")
                continue

            encoding = enc_list[0]
            print(f"\nNombre para la nueva cara (hay {db.get_face_count()} caras en DB):")
            name = input("  Nombre: ").strip()
            if not name:
                auto_count = db.get_face_count() + 1
                name = f"Persona_{auto_count}"

            face_id = db.save_face(name, encoding)
            print(f"  Guardado: {name} (ID {face_id})")
            known_faces = db.load_all_faces()

        elif key == ord('l'):
            print("\n--- Caras en base de datos ---")
            for f in db.list_faces():
                print(f"  [{f['id']}] {f['name']}  |  Guardado: {f['created_at'][:10]}  |  Avistamientos: {f['sightings']}")
            print()

    cap.release()
    cv2.destroyAllWindows()
    print("Face Tracker cerrado.")


if __name__ == "__main__":
    main()
