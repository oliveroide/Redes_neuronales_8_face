"""
App Streamlit de reconocimiento facial con cámara en vivo.

Ejecutar:
    streamlit run app.py

Recuadro verde  -> persona registrada (acceso autorizado)
Recuadro rojo   -> rostro desconocido  (acceso denegado)
"""

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from reconocedor import ReconocedorFacial, redimensionar

st.set_page_config(page_title="Reconocimiento facial", page_icon="🔐")


# ------------------------------------------------------------------ modelo + base de datos
@st.cache_resource(show_spinner="Cargando modelos y registrando rostros...")
def cargar_reconocedor():
    """Se carga una sola vez; la base de datos se lee de Fotos_para_FaceRecognition/."""
    rec = ReconocedorFacial()
    rec.cargar_carpeta()
    return rec


rec = cargar_reconocedor()


def dibujar_banner(img, resultados):
    """Franja superior con el veredicto: autorizado (verde) o denegado (rojo)."""
    if not resultados:
        return img
    autorizados = [r["nombre"] for r in resultados if r["autorizado"]]
    if len(autorizados) == len(resultados):
        texto, color = "ACCESO AUTORIZADO: " + ", ".join(autorizados), (60, 170, 60)
    else:
        texto, color = "DESCONOCIDO - ACCESO DENEGADO", (40, 40, 220)
    # El tamaño de letra se ajusta para que el texto quepa (el video del celular es vertical)
    ancho_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
    escala = min(0.8, (img.shape[1] - 20) / ancho_texto)
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), color, -1)
    cv2.putText(img, texto, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 2, cv2.LINE_AA)
    return img


# Las redes de OpenCV no son seguras entre hilos: cada sesión de video crea su propio
# reconocedor y copia la base de datos ya calculada (no recalcula los embeddings).
embeddings, nombres = rec.embeddings, rec.nombres


class ProcesadorRostros(VideoProcessorBase):
    def __init__(self):
        self.rec = ReconocedorFacial()
        self.rec.embeddings, self.rec.nombres = embeddings, nombres

    def recv(self, frame):
        img = redimensionar(frame.to_ndarray(format="bgr24"), 640)
        resultados = self.rec.reconocer(img)
        img = dibujar_banner(self.rec.dibujar(img, resultados), resultados)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ------------------------------------------------------------------ interfaz
st.title("🔐 Reconocimiento facial")
st.write("Presiona **START** y mira a la cámara. "
         "Recuadro **verde** = persona registrada, **rojo** = desconocido.")

webrtc_streamer(
    key="rostros",
    video_processor_factory=ProcesadorRostros,
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    async_processing=True,
)
