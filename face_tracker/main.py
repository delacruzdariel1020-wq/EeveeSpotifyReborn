#!/usr/bin/env python3
"""
EeveeFaceTracker – CLI entry point.

Usage examples
--------------
# Start webcam tracker (interactive)
python main.py live

# Process an image and show results
python main.py image foto.jpg

# Process an image and save the detected face as "Maria"
python main.py image foto.jpg --save "Maria" --output resultado.jpg

# List all saved faces
python main.py list

# Remove a saved face by name
python main.py remove "Maria"

# Show stored landmark features for a name
python main.py info "Maria"
"""

import argparse
import sys

from database import FaceDatabase
from tracker  import run_live, run_image


def cmd_live(args, db):
    run_live(db, show_landmarks=not args.no_landmarks, camera_index=args.camera)


def cmd_image(args, db):
    run_image(
        path=args.path,
        db=db,
        show_landmarks=not args.no_landmarks,
        save_name=args.save,
        output_path=args.output,
    )


def cmd_list(args, db):
    names = db.list_names()
    if not names:
        print("La base de datos está vacía.")
        return
    print(f"{len(db.faces)} entrada(s), {len(names)} nombre(s) únicos:\n")
    for name in names:
        entries = [f for f in db.faces if f["name"] == name]
        print(f"  {name!r}  ({len(entries)} muestra(s))")


def cmd_remove(args, db):
    n = db.remove_face(args.name)
    if n:
        print(f"Eliminadas {n} entrada(s) de '{args.name}'.")
    else:
        print(f"No se encontró '{args.name}' en la base de datos.")


def cmd_info(args, db):
    entries = [f for f in db.faces if f["name"] == args.name]
    if not entries:
        print(f"No se encontró '{args.name}'.")
        return
    for i, e in enumerate(entries):
        print(f"\n--- Muestra {i+1} de '{args.name}' (añadida {e['added_at']}) ---")
        lm = e.get("landmarks", {})
        if lm:
            for feat, val in lm.items():
                print(f"  {feat}: {val}")
        else:
            print("  (sin datos de landmarks guardados)")
        print(f"  fuente: {e.get('source') or '(webcam)'}")


def main():
    parser = argparse.ArgumentParser(
        prog="EeveeFaceTracker",
        description="Face tracker con almacenamiento de rasgos faciales.",
    )
    parser.add_argument(
        "--db", default=None,
        help="Ruta al archivo JSON de la base de datos (default: faces_db.json junto a este script).",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # live
    p_live = sub.add_parser("live", help="Abrir cámara web en tiempo real.")
    p_live.add_argument("--camera", type=int, default=0, help="Índice de la cámara (default: 0).")
    p_live.add_argument("--no-landmarks", action="store_true", help="Ocultar landmarks faciales.")

    # image
    p_img = sub.add_parser("image", help="Procesar una imagen estática.")
    p_img.add_argument("path", help="Ruta a la imagen.")
    p_img.add_argument("--save", metavar="NAME", help="Guardar la cara más grande con este nombre.")
    p_img.add_argument("--output", metavar="PATH", help="Guardar imagen anotada en esta ruta.")
    p_img.add_argument("--no-landmarks", action="store_true", help="Ocultar landmarks faciales.")

    # list
    sub.add_parser("list", help="Listar todas las caras guardadas.")

    # remove
    p_rm = sub.add_parser("remove", help="Eliminar una cara de la base de datos.")
    p_rm.add_argument("name", help="Nombre a eliminar.")

    # info
    p_info = sub.add_parser("info", help="Ver rasgos/landmarks de una cara guardada.")
    p_info.add_argument("name", help="Nombre a consultar.")

    args = parser.parse_args()

    db_path = args.db if args.db else None
    db = FaceDatabase(db_path) if db_path else FaceDatabase()

    dispatch = {
        "live":   cmd_live,
        "image":  cmd_image,
        "list":   cmd_list,
        "remove": cmd_remove,
        "info":   cmd_info,
    }
    dispatch[args.command](args, db)


if __name__ == "__main__":
    main()
