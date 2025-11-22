# src/classify.py
from transformers import pipeline
from PIL import Image
import os
import sys

# Ruta a la imagen dentro de /assets
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(HERE, "assets")
IMAGE_PATH = os.path.join(ASSETS_DIR, "perro.jpg")  # cambia el nombre si la llamaste distinto

def main(image_path=IMAGE_PATH):
    if not os.path.exists(image_path):
        print(f"ERROR: imagen no encontrada: {image_path}")
        sys.exit(1)

    # Carga pipeline pre-entrenado (ImageNet). Cambia el modelo si quieres uno especializado.
    clf = pipeline("image-classification", model="google/vit-base-patch16-224")

    # Clasificar
    results = clf(image_path)

    # Mostrar resultados
    print(f"Resultados para: {image_path}")
    for r in results:
        label = r.get("label", "<unknown>")
        score = r.get("score", 0.0)
        print(f" - {label}: {score:.4f}")

if __name__ == "__main__":
    main()
