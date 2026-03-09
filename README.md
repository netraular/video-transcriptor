# Video Transcriptor

This project generates `.srt` subtitle files from video files using two transcription modes:

- **Online mode** (`main_online.py`): Uses the **Groq API** (cloud) for fast transcription. Supports batch processing, folder selection, recursive search, language selection, SRT/TXT output, and MP3/MP4 input.
- **Local mode** (`main_local.py`): Uses **OpenAI's Whisper** model running entirely on your local machine. Supports single file selection from the `input/` folder and SRT output.

You can also run `main.py` which provides a launcher menu to choose between both modes.

## Prerequisites

1.  **Python 3.8 - 3.12**: Compatible with these Python versions. If you have multiple versions installed, ensure you use a compatible one for the virtual environment.
2.  **FFmpeg**: Required by `pydub` (and `whisper` in local mode) for handling audio/video files.
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    *   **Windows:** Download from the [official site](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.
3.  **GPU (Recommended for Local mode)**: While the local script can run on a CPU, transcribing with larger Whisper models (like `medium` or `large`) will be **extremely slow** without a GPU with sufficient VRAM.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd video-transcriptor
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    ```
    > **Note for users with multiple Python versions:** On Windows, you can use the `py` launcher:
    > ```bash
    > py -3.12 -m venv venv
    > ```

    Then, activate the environment:

    *   **On Windows (Command Prompt):**
        ```batch
        venv\Scripts\activate
        ```
    *   **On Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**

    *   **A) (For Local mode) Install PyTorch for your GPU:**
        Go to the [PyTorch official website](https://pytorch.org/get-started/locally/) for the correct command. Example for CUDA 12.1:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

    *   **B) Install all other requirements:**
        ```bash
        pip install -r requirements.txt
        ```

4.  **Set up your API Key (Online mode only):**
    *   Rename `.env.example` to `.env`.
    *   Add your Groq API key:
        ```ini
        GROQ_API_KEY="gsk_..."
        ```

## Usage

### Launcher (recommended)
```bash
python main.py
```
Choose between Online (Groq) or Local (Whisper) mode.

### Run directly
```bash
# Online mode
python main_online.py

# Local mode
python main_local.py
```

## Configuration

### Local mode
Edit the `MODEL` variable in `main_local.py`:
```python
MODEL = "medium"
# Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
```

### Online mode
Uses `whisper-large-v3` on Groq's servers. Language and output format (SRT/TXT) are configured interactively.