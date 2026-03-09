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
    Main function to let the user select a video file, then extract its audio,
    transcribe it locally with Whisper, and generate SRT captions.
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

    # --- 2. FILE SELECTION MENU ---
    # Find all available video files in the input directory
    video_files = [f for f in os.listdir(INPUT_DIR) if f.endswith((".mp4", ".mov", ".avi", ".mkv"))]

    if not video_files:
        print(f"Error: No video files (.mp4, .mov, etc.) found in the '{INPUT_DIR}' directory.")
        return

    # Display a numbered list of files for the user to choose from
    print("\nPlease choose a file to transcribe:")
    for i, filename in enumerate(video_files):
        print(f"  [{i+1}] {filename}")
    print()

    # Loop until the user provides a valid choice
    chosen_file = None
    while True:
        try:
            choice_str = input(f"Enter the number of the file (1-{len(video_files)}): ")
            choice_index = int(choice_str) - 1

            if 0 <= choice_index < len(video_files):
                chosen_file = video_files[choice_index]
                break
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(video_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            return

    # --- 3. PROCESSING THE CHOSEN FILE ---
    print(f"\n--- Processing selected file: {chosen_file} ---\n")

    input_path = os.path.join(INPUT_DIR, chosen_file)
    output_filename_base = os.path.splitext(chosen_file)[0]
    output_srt_path = os.path.join(OUTPUT_DIR, output_filename_base + ".srt")
    temp_audio_path = os.path.join(OUTPUT_DIR, "temp_audio.wav")

    # --- 4. AUDIO EXTRACTION & PREPARATION ---
    try:
        print("Step 1: Extracting audio from video...")
        audio = AudioSegment.from_file(input_path)

        print("Step 2: Preparing audio (converting to 16kHz mono WAV)...")
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_audio_path, format="wav")
        print("Temporary audio file created successfully.")
    except Exception as e:
        print(f"Error during audio processing for {chosen_file}: {e}")
        print("Please ensure FFmpeg is installed and accessible in your system's PATH.")
        return

    # --- 5. TRANSCRIPTION ---
    # Transcribe the entire audio file at once using local Whisper.
    print(f"Step 3: Transcribing audio using local Whisper model '{MODEL}'...")
    print("This may take a long time depending on the video length and your hardware (GPU is highly recommended).")

    try:
        # Set language to None to allow auto-detection, or specify e.g., "en", "es", "fr"
        transcription_result = model.transcribe(temp_audio_path, verbose=True, language="english")
    except Exception as e:
        print(f"    ! Error during transcription: {e}")
        os.remove(temp_audio_path)
        return

    print("Transcription complete.")

    # --- 6. SRT FILE GENERATION ---
    print(f"Step 4: Generating SRT file...")
    srt_content = ""
    for i, segment in enumerate(transcription_result['segments']):
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']

        srt_content += f"{i + 1}\n"
        srt_content += f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n"
        srt_content += f"{text.strip()}\n\n"

    # --- 7. SAVING THE SRT FILE & CLEANUP ---
    print(f"Step 5: Saving captions to {output_srt_path}")
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    # Clean up the temporary audio file
    os.remove(temp_audio_path)
    print("Temporary audio file deleted.")

    print(f"\n✅ Caption generation complete for {chosen_file}!")
    print(f"   Your SRT file is saved at: {output_srt_path}")

if __name__ == "__main__":
    main()
