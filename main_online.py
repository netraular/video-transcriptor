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
        return

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

    # --- 5. SAVING THE SRT FILE ---
    print(f"Step 5: Saving captions to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"\n✅ Caption generation complete for {input_path}!")
    print(f"   Your SRT file is saved at: {output_path}")

def main():
    """
    Main function to let the user select a folder, then process all mp4 files.
    """
    # --- 1. INITIALIZATION & SETUP ---
    print("Caption Generation Script Started (Online - Groq API)")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env file.")
        return
    client = Groq(api_key=api_key)
    print("Groq client initialized.")

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

    # --- FOLDER SELECTION ---
    default_dir = INPUT_DIR
    print(f"\nDefault directory: {default_dir}")
    use_default = input("Use default directory? (y/n): ").strip().lower()

    if use_default == 'y':
        target_dir = default_dir
    else:
        target_dir = input("Enter the directory path containing media files: ").strip()
        # Remove quotes if user copied path as "path"
        if target_dir.startswith('"') and target_dir.endswith('"'):
            target_dir = target_dir[1:-1]

    if not os.path.isdir(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    # --- RECURSIVE OPTION ---
    recursive = input("Search recursively in subfolders? (y/n): ").strip().lower() == 'y'

    # --- FILE TYPES ---
    extensions = {".mp4", ".flac", ".mp3", ".mpga", ".m4a", ".ogg", ".wav"}

    # --- FIND FILES ---
    video_files = []
    if recursive:
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in extensions:
                    video_files.append(os.path.join(root, file))
    else:
        video_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir)
                       if os.path.splitext(f)[1].lower() in extensions]

    if not video_files:
        print(f"No media files found in '{target_dir}'.")
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

        process_video(client, video_path, output_path, language, output_format)

    print("\nAll files processed.")

if __name__ == "__main__":
    main()
