"""
Pipeline automático: referencia -> notan (3 valores planos) -> maquette (mapa de bordes)

Uso:
    python3 notan_pipeline.py archivo1.png archivo2.jpg ...

Para cada imagen de entrada crea una carpeta con su nombre dentro de OUTPUT_DIR,
conteniendo:
    - reference.png   (recorte/ajuste al tamaño de maquette, tal cual)
    - notan.png        (3 valores planos: negro / gris medio / blanco)
    - maquette.png      (mapa de bordes del notan)
"""

import sys
import os
from PIL import Image, ImageOps
import numpy as np
import cv2

# ---- Parámetros configurables ----
TARGET_W_CM = 7
TARGET_H_CM = 5
DPI = 300
PERCENTILES = (33, 66)   # umbrales para los 3 valores del notan
BLUR_SIGMA = 6            # suavizado antes de segmentar en valores
CANNY_THRESHOLDS = (50, 150)
OUTPUT_DIR = "./outputs"
STRETCH_CONTRAST = False  # activar para fotos de bajo contraste (luz suave, tonos muy próximos)


def _stretch_contrast(gray_np: np.ndarray) -> np.ndarray:
    """Ecualización adaptativa de histograma (CLAHE) para separar mejor los valores
    tonales en imágenes de contraste bajo/plano, antes de segmentar en negro/gris/blanco."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray_np.astype(np.uint8)).astype(np.float32)


def process_image(path: str, output_dir: str = OUTPUT_DIR, stretch_contrast: bool = STRETCH_CONTRAST):
    name = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.join(output_dir, name)
    os.makedirs(folder, exist_ok=True)

    img = Image.open(path).convert("RGB")

    w, h = img.size

    # Orientacion de la maquette segun la orientacion de la imagen de entrada,
    # para no deformar imagenes verticales forzandolas a un marco apaisado.
    if w >= h:
        target_w = round(TARGET_W_CM / 2.54 * DPI)   # apaisado: 7x5
        target_h = round(TARGET_H_CM / 2.54 * DPI)
    else:
        target_w = round(TARGET_H_CM / 2.54 * DPI)   # vertical: 5x7
        target_h = round(TARGET_W_CM / 2.54 * DPI)
    target_ratio = target_w / target_h

    current_ratio = w / h
    if current_ratio > target_ratio:
        new_w = round(h * target_ratio)
        x0 = (w - new_w) // 2
        cropped = img.crop((x0, 0, x0 + new_w, h))
    else:
        new_h = round(w / target_ratio)
        y0 = (h - new_h) // 2
        cropped = img.crop((0, y0, w, y0 + new_h))

    reference = cropped.resize((target_w, target_h), Image.LANCZOS)
    reference.save(os.path.join(folder, "reference.png"), dpi=(DPI, DPI))

    # --- Notan ---
    gray = ImageOps.grayscale(reference)
    gray_np = np.array(gray, dtype=np.float32)
    if stretch_contrast:
        gray_np = _stretch_contrast(gray_np)
    blurred = cv2.GaussianBlur(gray_np, (0, 0), sigmaX=BLUR_SIGMA)

    p1, p2 = np.percentile(blurred, list(PERCENTILES))
    notan = np.zeros_like(blurred, dtype=np.uint8)
    notan[blurred <= p1] = 0
    notan[(blurred > p1) & (blurred <= p2)] = 128
    notan[blurred > p2] = 255

    notan_img = Image.fromarray(notan, mode="L")
    notan_img.save(os.path.join(folder, "notan.png"), dpi=(DPI, DPI))

    # --- Maquette (mapa de bordes del notan) ---
    edges = cv2.Canny(notan, *CANNY_THRESHOLDS)
    edges_inv = 255 - edges
    maquette_path = os.path.join(folder, "maquette.png")
    Image.fromarray(edges_inv, mode="L").save(maquette_path, dpi=(DPI, DPI))

    # --- Rejillas de composición sobre el notan ---
    notan_path = os.path.join(folder, "notan.png")
    reference_path = os.path.join(folder, "reference.png")
    for grid_type in ("thirds", "phi", "diagonal"):
        overlay_grid(
            notan_path,
            grid_type=grid_type,
            output_path=os.path.join(folder, f"notan_grid_{grid_type}.png"),
        )

    # --- Origen de luz sobre el notan ---
    mark_light_source(
        notan_path,
        reference_path,
        os.path.join(folder, "notan_light_source.png"),
    )

    return folder


def overlay_grid(image_path: str, grid_type: str = "thirds", output_path: str = None,
                  line_color=(255, 0, 0), line_width: int = 2):
    """Superpone una rejilla de composición fotográfica sobre una imagen (p.ej. el notan).

    grid_type:
        "thirds"   -> regla de tercios (rejilla 3x3 uniforme)
        "phi"      -> proporción áurea (líneas a 1/phi y 1-1/phi, ~38.2% y 61.8%)
        "spiral"   -> espiral áurea (Fibonacci), aproximada con arcos
        "diagonal" -> método de las diagonales (desde las 4 esquinas)
        "center"   -> cruz central simple
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw_img = img.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(draw_img)

    if grid_type == "thirds":
        for i in (1, 2):
            x = round(w * i / 3)
            draw.line([(x, 0), (x, h)], fill=line_color, width=line_width)
            y = round(h * i / 3)
            draw.line([(0, y), (w, y)], fill=line_color, width=line_width)

    elif grid_type == "phi":
        phi_inv = 0.618
        for frac in (1 - phi_inv, phi_inv):  # ~0.382 y ~0.618
            x = round(w * frac)
            draw.line([(x, 0), (x, h)], fill=line_color, width=line_width)
            y = round(h * frac)
            draw.line([(0, y), (w, y)], fill=line_color, width=line_width)

    elif grid_type == "diagonal":
        # las dos diagonales principales (forman una X de esquina a esquina)
        draw.line([(0, 0), (w, h)], fill=line_color, width=line_width)
        draw.line([(w, 0), (0, h)], fill=line_color, width=line_width)

    elif grid_type == "center":
        draw.line([(w // 2, 0), (w // 2, h)], fill=line_color, width=line_width)
        draw.line([(0, h // 2), (w, h // 2)], fill=line_color, width=line_width)

    elif grid_type == "spiral":
        # Aproximación de la espiral áurea con 4 cuartos de circunferencia
        # decreciendo según la proporción áurea, empezando en la esquina superior derecha.
        phi = 1.618
        size = min(w, h)
        # Definimos 4 cuadrados sucesivos y dibujamos un arco de 90 grados en cada uno.
        boxes = []
        cur_w, cur_h = w, h
        x0, y0 = 0, 0
        direction = 0  # 0=der,1=abajo,2=izq,3=arriba (rotación del cuadrado que se recorta)
        for _ in range(6):
            side = min(cur_w, cur_h) / phi
            if direction == 0:
                box = (x0 + cur_w - side, y0, x0 + cur_w, y0 + side)
                x0, cur_w = x0, cur_w - side
            elif direction == 1:
                box = (x0, y0 + cur_h - side, x0 + side, y0 + cur_h)
                cur_h = cur_h - side
            elif direction == 2:
                box = (x0, y0, x0 + side, y0 + side)
                x0, cur_w = x0 + side, cur_w - side
            else:
                box = (x0 + cur_w - side, y0, x0 + cur_w, y0 + side)
                y0, cur_h = y0 + side, cur_h - side
            boxes.append((box, direction))
            direction = (direction + 1) % 4

        for box, d in boxes:
            bx0, by0, bx1, by1 = box
            side = bx1 - bx0
            if side < 3:
                continue
            if d == 0:
                bbox = [bx0 - side, by0, bx1, by1 + side]
                start, end = 180, 270
            elif d == 1:
                bbox = [bx0 - side, by0 - side, bx1, by1]
                start, end = 90, 180
            elif d == 2:
                bbox = [bx0, by0 - side, bx1 + side, by1]
                start, end = 0, 90
            else:
                bbox = [bx0, by0, bx1 + side, by1 + side]
                start, end = 270, 360
            draw.arc(bbox, start=start, end=end, fill=line_color, width=line_width)

    else:
        raise ValueError(f"grid_type desconocido: {grid_type}")

    if output_path:
        draw_img.save(output_path, dpi=(DPI, DPI))
    return draw_img


def find_light_source(reference_path: str, blur_sigma: float = 15, top_percentile: float = 2.0):
    """Estima la posicion del foco de luz principal de una imagen: localiza el
    CENTRO DE MASA de la zona mas brillante (no un unico pixel), suavizando
    antes para ignorar reflejos puntuales pequenos.

    Usar el centro de masa (en vez del primer pixel maximo encontrado) evita
    que imagenes con grandes zonas de blanco puro empatado (ilustraciones de
    alto contraste, cielos sobreexpuestos, etc.) hagan que el punto caiga por
    error en la esquina superior izquierda (0,0), que es donde cv2.minMaxLoc
    reporta el primer empate en el orden de escaneo de la imagen.

    Devuelve (x, y) en pixeles.
    """
    gray = np.array(Image.open(reference_path).convert("L"), dtype=np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)

    threshold = np.percentile(blurred, 100 - top_percentile)
    bright_mask = blurred >= threshold

    ys, xs = np.nonzero(bright_mask)
    if len(xs) == 0:
        _, _, _, max_loc = cv2.minMaxLoc(blurred)
        return max_loc

    # centro de masa ponderado por brillo (no solo geometrico)
    weights = blurred[ys, xs] - blurred[ys, xs].min() + 1e-6
    x = int(np.average(xs, weights=weights))
    y = int(np.average(ys, weights=weights))
    return (x, y)


def mark_light_source(notan_path: str, reference_path: str, output_path: str,
                       circle_color=(255, 220, 0), radius: int = 18, width: int = 4):
    """Dibuja sobre el notan un círculo (+ cruz central) en la posición estimada
    de la fuente de luz principal, calculada a partir de la imagen de referencia."""
    x, y = find_light_source(reference_path)
    img = Image.open(notan_path).convert("RGB")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                 outline=circle_color, width=width)
    draw.line([(x - 6, y), (x + 6, y)], fill=circle_color, width=2)
    draw.line([(x, y - 6), (x, y + 6)], fill=circle_color, width=2)
    img.save(output_path, dpi=(DPI, DPI))
    return x, y


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 notan_pipeline.py <imagen1> [imagen2 ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        out = process_image(filepath)
        print(f"Procesada: {filepath} -> {out}")
