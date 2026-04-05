import cv2
import face_recognition
import numpy as np
import time
import threading
from face_db import FaceDatabase

# ── Config ───────────────────────────────────────────────────────────────
RECOGNITION_THRESHOLD = 0.45
SCAN_SAMPLES     = 15
NUM_JITTERS_LIVE = 1
NUM_JITTERS_SCAN = 10
SCALE            = 0.25
DB_PATH          = "faces_db.sqlite"

C_GREEN  = (0, 220, 0)
C_ORANGE = (0, 140, 255)
C_CYAN   = (0, 220, 220)
C_RED    = (0, 60, 220)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0, 0, 0)
C_DARK   = (20, 20, 20)

ST_NORMAL   = 0
ST_SCANNING = 1
ST_TYPING   = 2
ST_DELETE   = 3


# ── Hilo de reconocimiento ──────────────────────────────────────────────────

class RecognitionThread(threading.Thread):
    def __init__(self, db):
        super().__init__(daemon=True)
        self.db = db
        self._lock  = threading.Lock()
        self._frame = None
        self._results = []
        self._known   = db.load_all_faces()
        self._running = True
        self._smooth  = {}

    def feed(self, frame):
        with self._lock:
            self._frame = frame

    def results(self):
        with self._lock:
            return list(self._results)

    def reload(self):
        with self._lock:
            self._known = self.db.load_all_faces()
            self._smooth.clear()

    def stop(self):
        self._running = False

    @staticmethod
    def _lerp(a, b, t=0.35):
        return tuple(int(x + (y - x) * t) for x, y in zip(a, b))

    def run(self):
        while self._running:
            with self._lock:
                frame = self._frame
                known = list(self._known)

            if frame is None:
                time.sleep(0.003)
                continue

            small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs  = face_recognition.face_locations(rgb)
            encs  = face_recognition.face_encodings(rgb, locs, num_jitters=NUM_JITTERS_LIVE)
            known_encs = [f["encoding"] for f in known]
            out, seen = [], set()

            for loc, enc in zip(locs, encs):
                t2, r2, b2, l2 = [int(v / SCALE) for v in loc]
                raw  = (t2, r2, b2, l2)
                label, color, sim, fid = "Desconocido", C_ORANGE, None, None

                if known_encs:
                    d  = face_recognition.face_distance(known_encs, enc)
                    bi = int(np.argmin(d))
                    if d[bi] <= RECOGNITION_THRESHOLD:
                        m = known[bi]
                        label, color = m["name"], C_GREEN
                        sim, fid     = 1.0 - d[bi], m["id"]
                        self.db.log_sighting(fid, loc)

                key  = label if label != "Desconocido" else f"unk{len(seen)}"
                seen.add(key)
                bbox = self._lerp(self._smooth.get(key, raw), raw)
                self._smooth[key] = bbox
                out.append(dict(bbox=bbox, label=label, color=color, sim=sim, id=fid))

            active = {r["label"] if r["label"] != "Desconocido"
                      else f"unk{i}" for i, r in enumerate(out)}
            with self._lock:
                self._smooth  = {k: v for k, v in self._smooth.items() if k in active}
                self._results = out


# ── Helpers de dibujo ────────────────────────────────────────────────────────

def overlay_rect(frame, x1, y1, x2, y2, color=C_DARK, alpha=0.72):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    bg  = np.full_like(roi, color)
    frame[y1:y2, x1:x2] = cv2.addWeighted(bg, alpha, roi, 1 - alpha, 0)


def put(frame, text, x, y, scale=0.6, color=C_WHITE, thickness=1,
        font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_face(frame, bbox, label, color, sim):
    t, r, b, l = bbox
    cv2.rectangle(frame, (l, t), (r, b), color, 2, cv2.LINE_AA)
    for (x, y) in [(l, t), (r, t), (l, b), (r, b)]:
        dx = 14 if x == l else -14
        dy = 14 if y == t else -14
        cv2.line(frame, (x, y), (x + dx, y), color, 3, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + dy), color, 3, cv2.LINE_AA)
    txt = label + (f"  {sim:.0%}" if sim else "")
    tsz, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
    overlay_rect(frame, l, t - 28, l + tsz[0] + 10, t, color, alpha=0.9)
    put(frame, txt, l + 5, t - 8, 0.65, C_WHITE)


def draw_hud(frame, db_count, n):
    h, w = frame.shape[:2]
    overlay_rect(frame, 0, h - 40, w, h)
    put(frame, "S Guardar   D Eliminar   L Listar   Q/Esc Salir",
        10, h - 12, 0.47, C_WHITE)
    put(frame, f"DB: {db_count}   Detectadas: {n}", 10, 28, 0.62, C_WHITE, 2)


def draw_scan_progress(frame, prog, n):
    h, w = frame.shape[:2]
    cx, cy, r = w // 2, h // 2, 90
    overlay_rect(frame, cx - r - 20, cy - r - 20, cx + r + 20, cy + r + 40, alpha=0.65)
    cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, int(360 * prog), C_CYAN, 5, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r - 10, C_DARK, cv2.FILLED)
    for i, (txt, sc) in enumerate([("Escaneando...", 0.75), (f"{n} / {SCAN_SAMPLES}", 0.55)]):
        tsz, _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
        put(frame, txt, cx - tsz[0]//2, cy - 5 + i * 26, sc,
            C_CYAN if i == 0 else C_WHITE, 2, cv2.FONT_HERSHEY_SIMPLEX)


def draw_typing(frame, typed):
    h, w = frame.shape[:2]
    overlay_rect(frame, 0, h // 2 - 55, w, h // 2 + 58, alpha=0.88)
    put(frame, "Nombre de la persona:", 20, h // 2 - 18, 0.65, C_CYAN, 1,
        cv2.FONT_HERSHEY_SIMPLEX)
    cur = "_" if int(time.time() * 2) % 2 == 0 else " "
    put(frame, typed + cur, 20, h // 2 + 24, 0.9, C_WHITE, 2,
        cv2.FONT_HERSHEY_SIMPLEX)
    put(frame, "Enter confirmar   Esc cancelar", 20, h // 2 + 50,
        0.45, C_WHITE, 1, cv2.FONT_HERSHEY_SIMPLEX)


def draw_delete_menu(frame, faces, sel):
    h, w = frame.shape[:2]
    mh = min(len(faces) * 36 + 82, h - 40)
    my = (h - mh) // 2
    overlay_rect(frame, w // 2 - 290, my, w // 2 + 290, my + mh, alpha=0.92)
    put(frame, "Eliminar cara", w // 2 - 85, my + 28, 0.75, C_RED, 2,
        cv2.FONT_HERSHEY_SIMPLEX)
    for i, f in enumerate(faces):
        y   = my + 62 + i * 36
        bg  = C_RED if i == sel else C_DARK
        txt = f"[{f['id']}] {f['name']}   {f['sightings']} avistamientos"
        overlay_rect(frame, w // 2 - 280, y - 23, w // 2 + 280, y + 9, bg, 0.85)
        put(frame, txt, w // 2 - 268, y, 0.58, C_WHITE, 1, cv2.FONT_HERSHEY_SIMPLEX)
    put(frame, "Flechas arriba/abajo   Enter eliminar   Esc cancelar",
        w // 2 - 210, my + mh - 12, 0.43, C_WHITE, 1, cv2.FONT_HERSHEY_SIMPLEX)


# ── Escaneo ────────────────────────────────────────────────────────────────────

def scan_encoding(cap):
    encodings, start = [], time.time()
    while len(encodings) < SCAN_SAMPLES and (time.time() - start) < 12:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb)
        if locs:
            encs = face_recognition.face_encodings(rgb, locs[:1],
                                                    num_jitters=NUM_JITTERS_SCAN)
            if encs:
                encodings.append(encs[0])
        draw_scan_progress(frame, len(encodings) / SCAN_SAMPLES, len(encodings))
        cv2.imshow("Face Tracker", frame)
        cv2.waitKey(1)
    return np.mean(encodings, axis=0) if encodings else None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    db  = FaceDatabase(DB_PATH)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: no se pudo abrir la camara.")
        return

    cv2.namedWindow("Face Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Tracker", 1280, 720)

    rt = RecognitionThread(db)
    rt.start()

    state, typed_name, pending_enc = ST_NORMAL, "", None
    del_faces, del_sel = [], 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if state == ST_SCANNING:
            pass
        else:
            rt.feed(frame)
            res = rt.results()
            for r in res:
                draw_face(frame, r["bbox"], r["label"], r["color"], r["sim"])
            draw_hud(frame, db.get_face_count(), len(res))
            if state == ST_TYPING:
                draw_typing(frame, typed_name)
            elif state == ST_DELETE:
                draw_delete_menu(frame, del_faces, del_sel)

        cv2.imshow("Face Tracker", frame)
        key = cv2.waitKey(1) & 0xFF

        # Esc siempre cancela / sale
        if key == 27:
            if state != ST_NORMAL:
                state, typed_name, pending_enc = ST_NORMAL, "", None
            else:
                break

        elif state == ST_NORMAL:
            if key == ord('q'):
                break
            elif key == ord('s'):
                res = rt.results()
                if any(r["label"] == "Desconocido" for r in res):
                    state = ST_SCANNING
                    enc   = scan_encoding(cap)
                    if enc is not None:
                        pending_enc, typed_name, state = enc, "", ST_TYPING
                    else:
                        state = ST_NORMAL
            elif key == ord('d'):
                del_faces = db.list_faces()
                if del_faces:
                    del_sel, state = 0, ST_DELETE
            elif key == ord('l'):
                print("\n--- Caras guardadas ---")
                for f in db.list_faces():
                    print(f"  [{f['id']}] {f['name']}  "
                          f"{f['created_at'][:10]}  {f['sightings']} avist.")
                print()

        elif state == ST_TYPING:
            if key == 13:  # Enter
                name = typed_name.strip() or f"Persona_{db.get_face_count() + 1}"
                fid  = db.save_face(name, pending_enc)
                print(f"Guardado: {name} (ID {fid})")
                rt.reload()
                state, typed_name, pending_enc = ST_NORMAL, "", None
            elif key == 8:   # Backspace
                typed_name = typed_name[:-1]
            elif 32 <= key <= 126:
                typed_name += chr(key)

        elif state == ST_DELETE:
            if key in (82, 72, 119, 104):   # Up arrows / w / h
                del_sel = max(0, del_sel - 1)
            elif key in (84, 80, 115, 106): # Down arrows / s / j
                del_sel = min(len(del_faces) - 1, del_sel + 1)
            elif key == 13:  # Enter
                f   = del_faces[del_sel]
                db.delete_face(f["id"])
                print(f"Eliminado: {f['name']} (ID {f['id']})")
                rt.reload()
                state = ST_NORMAL

    rt.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
