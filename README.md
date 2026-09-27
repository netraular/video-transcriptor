# Video Transcriptor

Genera subtitulos `.srt` o texto `.txt` a partir de videos usando la API online de Groq.

## Requisitos

- Windows, macOS o Linux.
- Python 3.8-3.12. En Windows se recomienda Python 3.12.
- FFmpeg instalado y disponible en el `PATH`.
- Una clave de API de Groq.

## Instalacion en Windows

Desde PowerShell, dentro de la carpeta del proyecto:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Instala FFmpeg con [winget](https://www.gyan.dev/ffmpeg/builds/) o desde la [pagina oficial](https://ffmpeg.org/download.html). Comprueba la instalacion con:

```powershell
ffmpeg -version
```

En macOS y Linux, instala FFmpeg con el gestor de paquetes del sistema.

## Configuracion de Groq

Crea un archivo `.env` en la raiz del proyecto con:

```ini
GROQ_API_KEY="gsk_..."
```

No compartas ni subas este archivo al repositorio.

## Ejecucion

Activa el entorno virtual y ejecuta:

```powershell
.\venv\Scripts\Activate.ps1
python main_online.py
```

Selecciona las opciones en este orden para generar subtitulos:

1. `1` - Transcribe audio/video.
2. `en` - Idioma del audio, o el codigo que corresponda.
3. `1` - Salida SRT.
4. `1` - Archivo individual.
5. Introduce la ruta del video.

El archivo `.srt` se guarda en la misma carpeta que el video. Para procesar una carpeta completa, selecciona la opcion `2` como origen.

## Modo local

El modo local usa OpenAI Whisper en tu propio equipo. No necesita una clave de Groq, pero es mucho mas lento y requiere mas espacio y memoria.

### Instalacion adicional

El modo local no esta incluido en `requirements.txt`, porque las dependencias de Whisper y PyTorch son pesadas y no hacen falta para el modo online.

Con el entorno virtual activado, instala Whisper:

```powershell
pip install openai-whisper
```

Para usar una GPU NVIDIA, instala la version de PyTorch compatible con tu version de CUDA desde la [pagina oficial de PyTorch](https://pytorch.org/get-started/locally/). Por ejemplo:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Ejecucion

Coloca los videos en la carpeta `input` y ejecuta:

```powershell
.\venv\Scripts\Activate.ps1
python main_local.py
```

El script guarda los subtitulos en `output`. El modelo predeterminado es `medium`; puedes cambiarlo editando `MODEL` en `main_local.py`. Las opciones disponibles son `tiny`, `base`, `small`, `medium`, `large`, `large-v2` y `large-v3`.

Tambien puedes usar `python main.py` para abrir el menu y elegir entre el modo online y el local.