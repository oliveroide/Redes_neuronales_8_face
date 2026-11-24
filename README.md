# Reconocimiento facial — control de acceso con Streamlit

App web (funciona en celular y en computadora) que reconoce a las personas registradas en
`Fotos_para_FaceRecognition/` y marca como **DESCONOCIDO / acceso denegado** a cualquier
rostro que no esté en la base de datos.

## Archivos

| Archivo | Contenido |
|---|---|
| `reconocedor.py` | Núcleo del modelo: detección, alineación, embeddings, comparación y decisión |
| `app.py` | Interfaz Streamlit con la cámara en vivo |
| `index.html` | Página HTML que muestra el modelo e incrusta la app para interactuar con ella |
| `requirements.txt` | Dependencias |
| `Fotos_para_FaceRecognition/` | Base de datos de imágenes: **una foto por persona, el nombre del archivo es la etiqueta** |
| `modelos/` | Se crea solo; ahí se descargan los modelos `.onnx` la primera vez |

## Instalación

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`. La primera vez descarga los dos modelos de OpenCV Zoo
(~38 MB) en `modelos/`.

Después abre `index.html` en el navegador para ver la interfaz HTML con la app incrustada.

Prueba rápida sin interfaz (lista las personas registradas y reconoce las imágenes dadas):

```bash
python reconocedor.py ruta/a/una_foto.jpg
```

### Usarla desde el celular

La cámara del navegador sólo funciona con HTTPS (o en `localhost`). Con la app corriendo:

```bash
cloudflared tunnel --url http://localhost:8501
```

y abre en el celular la dirección `https://...trycloudflare.com` que muestra Cloudflare
(también se puede escribir en el campo **Servidor** de `index.html`). Es un túnel temporal:
cada vez que se reinicia `cloudflared` la dirección cambia.

`cloudflared` se descarga de la página oficial de Cloudflare
(<https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/>)
o en Windows con `winget install --id Cloudflare.cloudflared`.

## Cómo funciona

1. **Detección — YuNet**: CNN ligera que devuelve la caja del rostro y 5 puntos faciales
   (ojos, nariz, comisuras de la boca). Si una selfie está tan cerca que el rostro ocupa
   todo el cuadro, se agrega un borde y se reduce la imagen antes de detectar de nuevo.
2. **Alineación**: con los 5 puntos el rostro se rota/escala a 112×112 px.
3. **Embedding — SFace**: CNN entrenada para reconocimiento facial que convierte el rostro
   en un vector de 128 dimensiones (normalizado).
4. **Comparación**: similitud coseno entre el vector del rostro y los de la base de datos.
5. **Decisión**: si la mejor similitud es ≥ **0.42** se reconoce a esa persona; si no, el
   rostro es **desconocido**.

No hay entrenamiento: es *one-shot learning*, basta una foto por persona. Para mejorar la
precisión se pueden agregar más fotos de la misma persona (`ana.jpg`, `ana2.jpg`, …).

### Validación con las 14 fotos

- **Desconocidos (leave-one-out)**: se quitó a cada persona de la base y se intentó
  reconocerla con el resto. La mayor similitud entre dos personas distintas fue **0.382**
  (media 0.155), por debajo del umbral 0.42 → las 14 fueron rechazadas correctamente.
- **Variaciones de la misma persona**: cada foto se modificó (espejo, oscurecida, rotada 8°,
  baja resolución): **56/56** reconocidas correctamente.

## Registrar a una persona nueva

Copia su foto a `Fotos_para_FaceRecognition/` con su nombre (`nombre.jpg`) y reinicia la app.

## Guion sugerido para el video

1. Presionar **START** y pasar a 5 o más personas registradas → recuadro verde con su
   nombre y franja *ACCESO AUTORIZADO*.
2. Pasar a una persona que **no** esté en la carpeta → recuadro rojo *DESCONOCIDO* y franja
   *DESCONOCIDO - ACCESO DENEGADO*.
