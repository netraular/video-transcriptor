# Groq MP4 Caption Generator

This script uses the Groq API to generate `.srt` subtitle files for `.mp4` video files.

It's designed to handle long videos (10+ minutes) by automatically extracting the audio, splitting it into manageable chunks, and transcribing each chunk.

## Prerequisites

1.  **Python 3.8 - 3.12**: This project is compatible with Python versions from 3.8 up to 3.12.
2.  **FFmpeg**: The script uses the `pydub` library, which requires FFmpeg for handling audio/video files.
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    *   **Windows:** Download from the [official site](https://ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd groq-caption-generator
    ```

2.  **Create and Activate a Virtual Environment:**
    First, create the environment:
    ```bash
    python -m venv venv
    ```
    Then, activate it based on your operating system and terminal:

    *   **On macOS/Linux (bash/zsh):**
        ```bash
        source venv/bin/activate
        ```

    *   **On Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
        > **Note:** If you get an error in PowerShell about scripts being disabled, run this command once to allow scripts in your current session, then try activating again:
        > `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`

    *   **On Windows (Command Prompt - cmd.exe):**
        ```batch
        venv\Scripts\activate
        ```

3.  **Install dependencies:**
    Once your virtual environment is active, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    *   Rename the `.env.example` file to `.env`.
    *   Open the `.env` file and add your Groq API key:
        ```ini
        GROQ_API_KEY="gsk_..."
        ```

## Usage

1.  **Place your video file** inside the `input/` directory. For example: `input/my_long_video.mp4`.
2.  **Make sure your virtual environment is active**. You should see `(venv)` at the beginning of your terminal prompt.
3.  **Run the script:**
    ```bash
    python main.py
    ```
4.  The script will process the first `.mp4` file it finds in the `input` directory.
5.  A corresponding `.srt` file will be created in the `output/` directory (e.g., `output/my_long_video.srt`).