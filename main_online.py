import os
import io
import datetime
from groq import Groq
from dotenv import load_dotenv
from pydub import AudioSegment

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# Directories for input and output files
INPUT_DIR = r"./input"
OUTPUT_DIR = INPUT_DIR  # Save subtitles in the same directory

# Model to use for transcription
MODEL = "whisper-large-v3"

# Audio chunking settings (in milliseconds)
CHUNK_SIZE_MS = 5 * 60 * 1000  # 5 minutes
OVERLAP_MS = 5 * 1000         # 5 seconds

# Supported media file types
MEDIA_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".flac", ".mp3", ".mpga", ".m4a", ".ogg", ".wav"}


def normalize_input_path(path):
    """Normalize a user-provided path and strip wrapping quotes."""
    cleaned = path.strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    return cleaned


def collect_media_files(target_dir, recursive):
    """Collect media files from a directory, optionally including subfolders."""
    files_found = []

    if recursive:
        for root, _, files in os.walk(target_dir):
            for filename in files:
                if os.path.splitext(filename)[1].lower() in MEDIA_EXTENSIONS:
                    files_found.append(os.path.join(root, filename))
    else:
        files_found = [
            os.path.join(target_dir, filename)
            for filename in os.listdir(target_dir)
            if os.path.splitext(filename)[1].lower() in MEDIA_EXTENSIONS
        ]

    return sorted(files_found)


def get_api_keys():
    """Loads up to 3 Groq API keys from environment in priority order."""
    key_names = ["GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"]
    keys = []

    for key_name in key_names:
        key_value = os.environ.get(key_name)
        if key_value and key_value.strip():
            keys.append((key_name, key_value.strip()))

    return keys


def is_rate_limit_error(error):
    """Best-effort detection for API rate limit errors (HTTP 429)."""
    status_code = getattr(error, "status_code", None)
    message = str(error).lower()
    return (
        status_code == 429
        or "error code: 429" in message
        or "rate_limit_exceeded" in message
        or ("rate limit" in message and "429" in message)
    )

def format_srt_time(seconds):
    """Converts seconds (float) to SRT time format HH:MM:SS,ms"""
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def process_video(client, input_path, output_path, language="en", output_format="srt"):
    """
    Process a single video file: extract audio, transcribe, and save SRT or TXT.
    """
    print(f"\n--- Processing file: {input_path} ---\n")

    # --- 2. AUDIO EXTRACTION & PREPARATION ---
    try:
        print("Step 1: Extracting audio from file...")
        audio = AudioSegment.from_file(input_path)

        print("Step 2: Preparing audio (converting to 16kHz mono)...")
        audio = audio.set_frame_rate(16000).set_channels(1)
        print("Audio prepared successfully.")
    except Exception as e:
        print(f"Error during audio processing for {input_path}: {e}")
        print("Please ensure FFmpeg is installed and accessible in your system's PATH.")
        return "failed"

    # --- 3. AUDIO CHUNKING ---
    print("Step 3: Chunking audio for transcription...")
    chunks = []
    duration_ms = len(audio)
    for i in range(0, duration_ms, CHUNK_SIZE_MS):
        start_ms = i
        end_ms = min(i + CHUNK_SIZE_MS + OVERLAP_MS, duration_ms)
        chunk = audio[start_ms:end_ms]
        chunks.append((chunk, start_ms / 1000.0))

    print(f"Audio split into {len(chunks)} chunks.")

    # --- 4. TRANSCRIPTION & SRT GENERATION ---
    print(f"Step 4: Transcribing chunks using Groq model '{MODEL}'...")
    srt_content = ""
    caption_index = 1
    for i, (chunk, chunk_start_time_s) in enumerate(chunks):
        print(f"  - Transcribing chunk {i+1}/{len(chunks)}...")

        buffer = io.BytesIO()
        chunk.export(buffer, format="wav")
        buffer.seek(0)

        try:
            transcription = client.audio.transcriptions.create(
                file=("chunk.wav", buffer.read()),
                model=MODEL,
                response_format="verbose_json",
                language=language,
                timestamp_granularities=["segment"]
            )

            for segment in transcription.segments:
                if output_format == "srt":
                    start_time = chunk_start_time_s + segment['start']
                    end_time = chunk_start_time_s + segment['end']

                    if start_time < (chunk_start_time_s + (CHUNK_SIZE_MS / 1000.0)):
                        srt_content += f"{caption_index}\n"
                        srt_content += f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n"
                        srt_content += f"{segment['text'].strip()}\n\n"
                        caption_index += 1
                else:
                    # TXT mode: just append text
                    text = segment['text'].strip()
                    if text:
                        srt_content += f"{text}\n"

        except Exception as e:
            print(f"    ! Error transcribing chunk {i+1}: {e}")
            print("\n❌ Transcription failed. Output file was not generated.")
            if is_rate_limit_error(e):
                return "rate_limit"
            return "failed"

    # --- 5. SAVING THE SRT FILE ---
    print(f"Step 5: Saving captions to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"\n✅ Caption generation complete for {input_path}!")
    print(f"   Your SRT file is saved at: {output_path}")
    return "success"

def main():
    """
    Main function to let the user select a file or folder, then process media files.
    """
    # --- 1. INITIALIZATION & SETUP ---
    print("Caption Generation Script Started (Online - Groq API)")

    api_keys = get_api_keys()
    if not api_keys:
        print("Error: No API keys found in .env.")
        print("Add at least GROQ_API_KEY (and optionally GROQ_API_KEY_2, GROQ_API_KEY_3).")
        return

    current_key_index = 0
    current_key_name, current_key_value = api_keys[current_key_index]
    client = Groq(api_key=current_key_value)
    print(f"Groq client initialized with {current_key_name}.")

    # --- CONFIGURATION INPUTS ---
    language = input("Enter language code (default 'en'): ").strip()
    if not language:
        language = "en"

    # 2. Output Format
    print("\nSelect output format:")
    print("1. Subtitles (SRT)")
    print("2. Plain Text (TXT)")
    format_choice = input("Choice [1]: ").strip()

    if format_choice == '2':
        output_format = 'txt'
    else:
        output_format = 'srt'

    # --- SOURCE SELECTION ---
    print("\nSelect input source:")
    print("1. Single file")
    print("2. Folder")
    source_choice = input("Choice [2]: ").strip()
    if not source_choice:
        source_choice = "2"

    folder_mode = source_choice != "1"
    only_missing_transcriptions = False

    video_files = []
    if source_choice == "1":
        file_path = normalize_input_path(input("Enter media file path: "))

        if not os.path.isfile(file_path):
            print(f"Error: File '{file_path}' does not exist.")
            return

        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in MEDIA_EXTENSIONS:
            print(f"Error: Unsupported file extension '{file_extension}'.")
            print("Supported extensions: " + ", ".join(sorted(MEDIA_EXTENSIONS)))
            return

        video_files = [file_path]
    else:
        default_dir = INPUT_DIR
        print(f"\nDefault directory: {default_dir}")
        use_default = input("Use default directory? (y/n): ").strip().lower()

        if use_default == 'y':
            target_dir = default_dir
        else:
            target_dir = normalize_input_path(input("Enter the directory path containing media files: "))

        if not os.path.isdir(target_dir):
            print(f"Error: Directory '{target_dir}' does not exist.")
            return

        recursive = input("Search recursively in subfolders? (y/n): ").strip().lower() == 'y'
        video_files = collect_media_files(target_dir, recursive)

    if folder_mode:
        only_missing_transcriptions = (
            input("Process only files without existing transcription? (y/n): ").strip().lower() == "y"
        )

        if only_missing_transcriptions:
            output_ext = ".srt" if output_format == "srt" else ".txt"
            files_before_filter = len(video_files)
            video_files = [
                file_path
                for file_path in video_files
                if not os.path.exists(
                    os.path.join(
                        os.path.dirname(file_path),
                        os.path.splitext(os.path.basename(file_path))[0] + output_ext,
                    )
                )
            ]
            skipped_existing = files_before_filter - len(video_files)
            if skipped_existing > 0:
                print(f"Skipping {skipped_existing} file(s) with existing transcription.")

    if not video_files:
        print("No supported media files found for processing.")
        return

    print(f"\nFound {len(video_files)} media files.")

    # --- PROCESS FILES ---
    for i, video_path in enumerate(video_files):
        print(f"\n[{i+1}/{len(video_files)}] Processing: {video_path}")

        # Determine output path (same directory as video)
        output_ext = ".srt" if output_format == "srt" else ".txt"
        output_filename = os.path.splitext(os.path.basename(video_path))[0] + output_ext
        output_path = os.path.join(os.path.dirname(video_path), output_filename)

        if os.path.exists(output_path):
            overwrite = input(f"Output file already exists for {os.path.basename(video_path)}. Overwrite? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("Skipping...")
                continue

        while True:
            result = process_video(client, video_path, output_path, language, output_format)

            if result == "success":
                break

            if result == "rate_limit":
                if current_key_index + 1 >= len(api_keys):
                    print("No more API keys available after rate limit error. Stopping process.")
                    return

                current_key_index += 1
                current_key_name, current_key_value = api_keys[current_key_index]
                client = Groq(api_key=current_key_value)
                print(f"Switched to {current_key_name}. Retrying current file from scratch...")
                continue

            print("Skipping this file due to transcription failure.")
            break

    print("\nAll files processed.")

if __name__ == "__main__":
    main()
