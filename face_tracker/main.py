import cv2
import face_recognition
import numpy as np
import time
import threading
from face_db import FaceDatabase

# --- Config ---
RECOGNITION_THRESHOLD = 0.45
SCAN_SAMPLES     = 15      # muestras al guardar
NUM_JITTERS_LIVE = 1       # rapido para el video en vivo
NUM_JITTERS_SCAN = 10      # preciso al escanear para guardar
SCALE            = 0.25    # escalar mas pequeno = deteccion mas rapida
DB_PATH          = "faces_db.sqlite"

COLOR_KNOWN   = (0, 220, 0)
COLOR_UNKNOWN = (0, 100, 255)
COLOR_TEXT    = (255, 255, 255)
COLOR_SCAN    = (0, 220, 220)
COLOR_DELETE  = (0, 60, 220)


# ── Hilo de reconocimiento ──────────────────────────────────────────────────

class RecognitionThread(threading.Thread):
    def __init__(self, db):
        super().__init__(daemon=True)
        self.db = db
        self._lock = threading.Lock()
        self._frame = None
        self._results = []
        self._known_faces = db.load_all_faces()
        self._stop_evt = threading.Event()
        self._prev_bboxes = {}

    def update_frame(self, frame):
        with self._lock:
            self._frame = frame

    def get_results(self):
        with self._lock:
            return list(self._results)

    def reload_faces(self):
        with self._lock:
            self._known_faces = self.db.load_all_faces()

    def stop(self):
        self._stop_evt.set()

    def _lerp_bbox(self, prev, curr, alpha=0.4):
        if prev is None:
            return curr
        return tuple(int(p + (c - p) * alpha) for p, c in zip(prev, curr))

    def run(self):
        while not self._stop_evt.is_set():
            with self._lock:
                frame = self._frame
                known = list(self._known_faces)

            if frame is None:
                time.sleep(0.005)
                continue

            small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locs  = face_recognition.face_locations(rgb)
            encs  = face_recognition.face_encodings(rgb, locs, num_jitters=NUM_JITTERS_LIVE)

            known_encs = [f["encoding"] for f in known]
            results    = []
            seen_names = set()

            for loc, enc in zip(locs, encs):
                top, right, bottom, left = [int(v / SCALE) for v in loc]
                raw_bbox = (top, right, bottom, left)

                label = "Desconocido"
                color = COLOR_UNKNOWN
                sim   = None
                fid   = None

                if known_encs:
                    dists    = face_recognition.face_distance(known_encs, enc)
                    best_idx = int(np.argmin(dists))
                    best_d   = dists[best_idx]
                    if best_d <= RECOGNITION_THRESHOLD:
                        m     = known[best_idx]
                        label = m["name"]
                        color = COLOR_KNOWN
                        sim   = 1.0 - best_d
                        fid   = m["id"]
                        self.db.log_sighting(m["id"], loc)

                key  = label if label != "Desconocido" else f"unk_{len(seen_names)}"
                seen_names.add(key)
                bbox = self._lerp_bbox(self._prev_bboxes.get(key), raw_bbox)
                self._prev_bboxes[key] = bbox

                results.append({"bbox": bbox, "label": label,
                                 "color": color, "sim": sim, "id": fid})

            active = {r["label"] if r["label"] != "Desconocido"
                      else f"unk_{i}" for i, r in enumerate(results)}
            self._prev_bboxes = {k: v for k, v in self._prev_bboxes.items()
                                 if k in active}

            with self._lock:
                self._results = results


# ── Dibujo ──────────────────────────────────────────────────────────────────

def draw_bbox(frame, bbox, label, color, sim=None):
    top, right, bottom, left = bbox
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    sz = 12
    for (x, y) in [(left, top), (right, top), (left, bottom), (right, bottom)]:
        dx = sz if x == left else -sz
        dy = sz if y == top  else -sz
        cv2.line(frame, (x, y), (x + dx, y), color, 3)
        cv2.line(frame, (x, y), (x, y + dy), color, 3)

    text = label + (f"  {sim:.0%}" if sim is not None else "")
    sz2, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    cv2.rectangle(frame, (left, top - 26), (left + sz2[0] + 8, top), color, cv2.FILLED)
    cv2.putText(frame, text, (left + 4, top - 7),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_TEXT, 1)


def draw_scan_overlay(frame, progress, samples):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    r = 100
    cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, int(360 * progress), COLOR_SCAN, 5)
    cv2.circle(frame, (cx, cy), r - 10, (0, 0, 0), cv2.FILLED)
    t1 = "Escaneando..."
    t2 = f"{samples}/{SCAN_SAMPLES} muestras"
    for i, (t, sc, th) in enumerate([(t1, 0.75, 2), (t2, 0.52, 1)]):
        sz, _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
        y = cy - 10 + i * 28
        cv2.putText(frame, t, (cx - sz[0]//2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sc,
                    COLOR_SCAN if i == 0 else COLOR_TEXT, th)


def draw_hud(frame, db_count, detected):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - 44), (w, h), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "  [S] Guardar  [D] Eliminar  [L] Listar  [Q] Salir",
                (10, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT, 1)
    cv2.putText(frame, f"DB: {db_count} caras  |  En camara: {detected}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)


# ── Escaneo multi-muestra ────────────────────────────────────────────────────

def scan_face(cap):
    encodings = []
    start = time.time()
    timeout = 10.0
    while len(encodings) < SCAN_SAMPLES and (time.time() - start) < timeout:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb)
        if locs:
            encs = face_recognition.face_encodings(rgb, locs[:1], num_jitters=NUM_JITTERS_SCAN)
            if encs:
                encodings.append(encs[0])
        prog    = len(encodings) / SCAN_SAMPLES
        display = frame.copy()
        draw_scan_overlay(display, prog, len(encodings))
        cv2.imshow("Face Tracker", display)
        cv2.waitKey(1)

    return np.mean(encodings, axis=0) if encodings else None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    db  = FaceDatabase(DB_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: no se pudo abrir la camara.")
        return

    print("Face Tracker iniciado.")
    print("  S -> guardar cara  |  D -> eliminar cara  |  L -> listar  |  Q -> salir\n")

    rt = RecognitionThread(db)
    rt.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rt.update_frame(frame)
        results = rt.get_results()

        for r in results:
            draw_bbox(frame, r["bbox"], r["label"], r["color"], r["sim"])

        draw_hud(frame, db.get_face_count(), len(results))
        cv2.imshow("Face Tracker", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            unknown = [r for r in results if r["label"] == "Desconocido"]
            if not unknown:
                print("No hay caras desconocidas en pantalla.")
                continue
            print(f"\nEscaneando... ({SCAN_SAMPLES} muestras, mantente quieto)")
            enc = scan_face(cap)
            if enc is None:
                print("  No se pudo escanear. Intentalo de nuevo.")
                continue
            print(f"Nombre (DB tiene {db.get_face_count()} caras):")
            name = input("  Nombre: ").strip() or f"Persona_{db.get_face_count() + 1}"
            fid  = db.save_face(name, enc)
            print(f"  Guardado: {name} (ID {fid})\n")
            rt.reload_faces()

        elif key == ord('d'):
            faces = db.list_faces()
            if not faces:
                print("No hay caras guardadas.")
                continue
            print("\n--- Caras guardadas ---")
            for f in faces:
                print(f"  [{f['id']}] {f['name']}  |  {f['created_at'][:10]}  |  {f['sightings']} avistamientos")
            try:
                fid = int(input("  ID a eliminar (Enter para cancelar): ").strip())
                db.delete_face(fid)
                print(f"  Cara ID {fid} eliminada.\n")
                rt.reload_faces()
            except (ValueError, EOFError):
                print("  Cancelado.\n")

        elif key == ord('l'):
            print("\n--- Caras guardadas ---")
            for f in db.list_faces():
                print(f"  [{f['id']}] {f['name']}  |  {f['created_at'][:10]}  |  {f['sightings']} avistamientos")
            print()

    rt.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Face Tracker cerrado.")


if __name__ == "__main__":
    main()
