"""
Núcleo del reconocimiento facial (sin interfaz).

Pipeline:
    1. Detección      -> YuNet  (red convolucional ligera, OpenCV Zoo, ONNX)
    2. Alineación     -> FaceRecognizerSF.alignCrop usa los 5 puntos faciales de YuNet
    3. Embedding      -> SFace  (CNN entrenada con pérdida SFace, vector de 128 dimensiones)
    4. Comparación    -> similitud coseno contra la base de datos de rostros registrados
    5. Decisión       -> si la mejor similitud >= umbral -> persona autorizada,
                         si no -> "Desconocido" (acceso denegado)

La base de datos es simplemente la carpeta `Fotos_para_FaceRecognition/`:
el nombre de cada archivo (sin extensión ni números) es el nombre de la persona.
Se pueden tener varias fotos por persona: `ana.jpg`, `ana2.jpg`, `ana_3.png` -> "Ana".
"""

import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ------------------------------------------------------------------ rutas
BASE = Path(__file__).resolve().parent
CARPETA_FOTOS = BASE / "Fotos_para_FaceRecognition"
CARPETA_MODELOS = BASE / "modelos"

# Modelos pre-entrenados oficiales de OpenCV Zoo (se descargan una sola vez)
URL_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELOS = {
    "face_detection_yunet_2023mar.onnx": f"{URL_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{URL_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

# Umbral de similitud coseno recomendado por los autores de SFace (0.363).
# Lo subimos a 0.42: en nuestras 14 fotos la mayor similitud entre personas
# distintas fue 0.382, así que 0.42 deja margen para rechazar desconocidos.
UMBRAL_COSENO = 0.42

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def descargar_modelos():
    """Descarga los .onnx de OpenCV Zoo si aún no existen en `modelos/`."""
    CARPETA_MODELOS.mkdir(exist_ok=True)
    for nombre, url in MODELOS.items():
        destino = CARPETA_MODELOS / nombre
        if not destino.exists():
            print(f"Descargando {nombre} ...")
            urllib.request.urlretrieve(url, destino)
    return [CARPETA_MODELOS / n for n in MODELOS]


def nombre_desde_archivo(ruta: Path) -> str:
    """'Edelmira1.png' -> 'Edelmira';  'juan_perez_2.jpg' -> 'Juan Perez'."""
    nombre = re.sub(r"[\d_\-]+$", "", ruta.stem)          # quita números finales
    nombre = nombre.replace("_", " ").replace("-", " ").strip()
    return nombre.title() if nombre else ruta.stem


def leer_imagen(ruta: Path) -> np.ndarray:
    """cv2.imread falla con rutas con acentos/espacios en Windows; imdecode no."""
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def redimensionar(img: np.ndarray, lado_max: int = 1280) -> np.ndarray:
    """Reduce fotos enormes (p. ej. 4928x6560) para que la detección sea rápida."""
    h, w = img.shape[:2]
    escala = lado_max / max(h, w)
    if escala < 1:
        img = cv2.resize(img, (int(w * escala), int(h * escala)), interpolation=cv2.INTER_AREA)
    return img


class ReconocedorFacial:
    def __init__(self, umbral: float = UMBRAL_COSENO, conf_deteccion: float = 0.8):
        ruta_det, ruta_rec = descargar_modelos()
        self.detector = cv2.FaceDetectorYN.create(str(ruta_det), "", (320, 320), conf_deteccion, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(ruta_rec), "")
        self.umbral = umbral
        self.nombres: list[str] = []          # un nombre por embedding
        self.embeddings = np.empty((0, 128), dtype=np.float32)
        self.archivos: dict[str, list[Path]] = {}   # persona -> fotos usadas

    # -------------------------------------------------------------- detección
    def detectar(self, img: np.ndarray) -> np.ndarray:
        """Devuelve un arreglo (N, 15): x, y, w, h, 10 coords de landmarks, score."""
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, caras = self.detector.detect(img)
        return caras if caras is not None else np.empty((0, 15), dtype=np.float32)

    def detectar_robusto(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
        """
        Detecta caras y, si no encuentra ninguna, reintenta con la imagen "alejada":
        YuNet falla con selfies donde la cara ocupa todo el cuadro, así que se agrega
        un borde y se reduce. Devuelve (imagen_usada, caras, escala, borde) para que
        los landmarks coincidan con la imagen que luego se alinea y para poder
        regresar las cajas a coordenadas de la imagen original.
        """
        caras = self.detectar(img)
        if len(caras):
            return img, caras, 1.0, 0
        borde = int(0.4 * max(img.shape[:2]))
        alejada = cv2.copyMakeBorder(img, borde, borde, borde, borde, cv2.BORDER_REPLICATE)
        escala = min(1.0, 640 / max(alejada.shape[:2]))
        alejada = redimensionar(alejada, 640)
        return alejada, self.detectar(alejada), escala, borde

    def embedding(self, img: np.ndarray, cara: np.ndarray) -> np.ndarray:
        """Alinea la cara a 112x112 con los landmarks y obtiene su vector SFace normalizado."""
        alineada = self.recognizer.alignCrop(img, cara)
        vec = self.recognizer.feature(alineada).flatten()
        return vec / np.linalg.norm(vec)

    # -------------------------------------------------------------- base de datos
    def registrar(self, img: np.ndarray, nombre: str, ruta: Path | None = None) -> bool:
        """Agrega la cara más grande de `img` a la base de datos. False si no hay cara."""
        img, caras, _, _ = self.detectar_robusto(redimensionar(img))
        if len(caras) == 0:
            return False
        cara = max(caras, key=lambda c: c[2] * c[3])     # la cara más grande
        self.embeddings = np.vstack([self.embeddings, self.embedding(img, cara)])
        self.nombres.append(nombre)
        if ruta is not None:
            self.archivos.setdefault(nombre, []).append(ruta)
        return True

    def cargar_carpeta(self, carpeta: Path = CARPETA_FOTOS) -> list[str]:
        """Registra todas las fotos de la carpeta. Devuelve los archivos sin rostro detectado."""
        self.nombres, self.archivos = [], {}
        self.embeddings = np.empty((0, 128), dtype=np.float32)
        fallidos = []
        for ruta in sorted(carpeta.iterdir()):
            if ruta.suffix.lower() not in EXTENSIONES:
                continue
            img = leer_imagen(ruta)
            if img is None or not self.registrar(img, nombre_desde_archivo(ruta), ruta):
                fallidos.append(ruta.name)
        return fallidos

    @property
    def personas(self) -> list[str]:
        return sorted(set(self.nombres))

    # -------------------------------------------------------------- reconocimiento
    def identificar(self, vec: np.ndarray) -> tuple[str, float]:
        """Compara contra todos los embeddings (producto punto = coseno, ya normalizados)."""
        if len(self.nombres) == 0:
            return "Desconocido", 0.0
        sims = self.embeddings @ vec
        i = int(np.argmax(sims))
        sim = float(sims[i])
        return (self.nombres[i] if sim >= self.umbral else "Desconocido"), sim

    def reconocer(self, img: np.ndarray) -> list[dict]:
        """Detecta y reconoce todas las caras de una imagen BGR."""
        resultados = []
        usada, caras, escala, borde = self.detectar_robusto(img)
        for cara in caras:
            nombre, sim = self.identificar(self.embedding(usada, cara))
            # Regresar la caja a coordenadas de la imagen original (si se usó la versión alejada)
            x, y, w, h = (cara[:4] / escala).astype(int)
            x, y = x - borde, y - borde
            resultados.append({"nombre": nombre, "similitud": sim, "caja": (x, y, w, h),
                               "autorizado": nombre != "Desconocido"})
        return resultados

    @staticmethod
    def dibujar(img: np.ndarray, resultados: list[dict]) -> np.ndarray:
        """Dibuja recuadros: verde = autorizado, rojo = desconocido."""
        out = img.copy()
        grosor = max(2, img.shape[1] // 400)
        escala = max(0.6, img.shape[1] / 1200)
        for r in resultados:
            x, y, w, h = r["caja"]
            color = (60, 200, 60) if r["autorizado"] else (40, 40, 230)
            etiqueta = f'{r["nombre"]} ({r["similitud"]:.2f})' if r["autorizado"] else "DESCONOCIDO"
            cv2.rectangle(out, (x, y), (x + w, y + h), color, grosor)
            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
            cv2.rectangle(out, (x, y - th - 12), (x + tw + 8, y), color, -1)
            cv2.putText(out, etiqueta, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, escala,
                        (255, 255, 255), grosor, cv2.LINE_AA)
        return out


if __name__ == "__main__":
    # Prueba rápida desde la terminal: python reconocedor.py [imagen]
    import sys

    rec = ReconocedorFacial()
    fallidos = rec.cargar_carpeta()
    print(f"Personas registradas ({len(rec.personas)}): {', '.join(rec.personas)}")
    if fallidos:
        print("Sin rostro detectado:", fallidos)
    for arg in sys.argv[1:]:
        img = redimensionar(leer_imagen(Path(arg)))
        for r in rec.reconocer(img):
            print(f"{arg}: {r['nombre']}  similitud={r['similitud']:.3f}")
