import os
import datetime
import whisper
from pydub import AudioSegment

# --- CONFIGURATION ---
# Directories for input and output files
INPUT_DIR = "input"
OUTPUT_DIR = "output"

# Model to use for transcription.
# Options: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3"
# Larger models are more accurate but require more VRAM and are slower.
MODEL = "medium"

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

# --- HELPER FUNCTION ---
def format_srt_time(seconds):
    """Converts seconds (float) to SRT time format HH:MM:SS,ms"""
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def main():
    """
    Main function to let the user select a media file or folder, then extract audio,
    transcribe locally with Whisper, and generate SRT captions.
    """
    # --- 1. INITIALIZATION & SETUP ---
    print("Local Caption Generation Script Started")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    # Load the Whisper model locally
    # This will download the model on the first run.
    print(f"Loading Whisper model '{MODEL}'... (This may take a moment)")
    try:
        model = whisper.load_model(MODEL)
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        print("Please ensure you have a valid model name and that you have enough memory.")
        return

    # --- 2. SOURCE SELECTION ---
    print("\nSelect input source:")
    print("1. Single file")
    print("2. Folder")
    source_choice = input("Choice [2]: ").strip()
    if not source_choice:
        source_choice = "2"

    folder_mode = source_choice != "1"
    only_missing_transcriptions = False

    media_files = []
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

        media_files = [file_path]
    else:
        default_dir = INPUT_DIR
        print(f"\nDefault directory: {default_dir}")
        use_default = input("Use default directory? (y/n): ").strip().lower()

        if use_default == "y":
            target_dir = default_dir
        else:
            target_dir = normalize_input_path(input("Enter the directory path containing media files: "))

        if not os.path.isdir(target_dir):
            print(f"Error: Directory '{target_dir}' does not exist.")
            return

        recursive = input("Search recursively in subfolders? (y/n): ").strip().lower() == "y"
        media_files = collect_media_files(target_dir, recursive)

    if folder_mode:
        only_missing_transcriptions = (
            input("Process only files without existing transcription? (y/n): ").strip().lower() == "y"
        )

        if only_missing_transcriptions:
            files_before_filter = len(media_files)
            media_files = [
                file_path
                for file_path in media_files
                if not os.path.exists(
                    os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(file_path))[0] + ".srt")
                )
            ]
            skipped_existing = files_before_filter - len(media_files)
            if skipped_existing > 0:
                print(f"Skipping {skipped_existing} file(s) with existing transcription.")

    if not media_files:
        print("No supported media files found for processing.")
        return

    print(f"\nFound {len(media_files)} media files.")

    # --- 3. PROCESS FILES ---
    for file_index, input_path in enumerate(media_files):
        chosen_file = os.path.basename(input_path)
        print(f"\n[{file_index + 1}/{len(media_files)}] Processing: {input_path}")

        output_filename_base = os.path.splitext(chosen_file)[0]
        output_srt_path = os.path.join(OUTPUT_DIR, output_filename_base + ".srt")
        temp_audio_path = os.path.join(OUTPUT_DIR, f"temp_audio_{file_index}.wav")

        if os.path.exists(output_srt_path):
            overwrite = input(f"Output file already exists for {chosen_file}. Overwrite? (y/n): ").strip().lower()
            if overwrite != "y":
                print("Skipping...")
                continue

        # --- 4. AUDIO EXTRACTION & PREPARATION ---
        try:
            print("Step 1: Extracting audio from media file...")
            audio = AudioSegment.from_file(input_path)

            print("Step 2: Preparing audio (converting to 16kHz mono WAV)...")
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(temp_audio_path, format="wav")
            print("Temporary audio file created successfully.")
        except Exception as e:
            print(f"Error during audio processing for {chosen_file}: {e}")
            print("Please ensure FFmpeg is installed and accessible in your system's PATH.")
            continue

        # --- 5. TRANSCRIPTION ---
        # Transcribe the entire audio file at once using local Whisper.
        print(f"Step 3: Transcribing audio using local Whisper model '{MODEL}'...")
        print("This may take a long time depending on the media length and your hardware (GPU is highly recommended).")

        try:
            # Set language to None to allow auto-detection, or specify e.g., "en", "es", "fr"
            transcription_result = model.transcribe(temp_audio_path, verbose=True, language="english")
        except Exception as e:
            print(f"    ! Error during transcription: {e}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            continue

        print("Transcription complete.")

        # --- 6. SRT FILE GENERATION ---
        print("Step 4: Generating SRT file...")
        srt_content = ""
        for i, segment in enumerate(transcription_result["segments"]):
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]

            srt_content += f"{i + 1}\n"
            srt_content += f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n"
            srt_content += f"{text.strip()}\n\n"

        # --- 7. SAVING THE SRT FILE & CLEANUP ---
        print(f"Step 5: Saving captions to {output_srt_path}")
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print("Temporary audio file deleted.")

        print(f"\n✅ Caption generation complete for {chosen_file}!")
        print(f"   Your SRT file is saved at: {output_srt_path}")

    print("\nAll files processed.")

if __name__ == "__main__":
    main()
