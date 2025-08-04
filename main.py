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
INPUT_DIR = "input"
OUTPUT_DIR = "output"

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

def main():
    """
    Main function to let the user select a video file, then extract its audio,
    chunk it, transcribe it, and generate SRT captions.
    """
    # --- 1. INITIALIZATION & SETUP ---
    print("Caption Generation Script Started")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env file.")
        return
    client = Groq(api_key=api_key)
    print("Groq client initialized.")

    # --- MODIFICATION: FILE SELECTION MENU ---
    # Find all available .mp4 files in the input directory
    video_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mp4")]

    if not video_files:
        print(f"Error: No .mp4 files found in the '{INPUT_DIR}' directory.")
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
            choice_index = int(choice_str) - 1  # Convert to 0-based index

            if 0 <= choice_index < len(video_files):
                chosen_file = video_files[choice_index]
                break  # Exit the loop with a valid choice
            else:
                print(f"Invalid number. Please enter a number between 1 and {len(video_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            return

    # --- PROCESSING THE CHOSEN FILE ---
    print(f"\n--- Processing selected file: {chosen_file} ---\n")
    
    input_path = os.path.join(INPUT_DIR, chosen_file)
    output_filename = os.path.splitext(chosen_file)[0] + ".srt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # --- 2. AUDIO EXTRACTION & PREPARATION ---
    try:
        print("Step 1: Extracting audio from video...")
        audio = AudioSegment.from_file(input_path, "mp4")
        
        print("Step 2: Preparing audio (converting to 16kHz mono)...")
        audio = audio.set_frame_rate(16000).set_channels(1)
        print("Audio prepared successfully.")
    except Exception as e:
        print(f"Error during audio processing for {chosen_file}: {e}")
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
                language="en",
                timestamp_granularities=["segment"]
            )
            
            for segment in transcription.segments:
                start_time = chunk_start_time_s + segment['start']
                end_time = chunk_start_time_s + segment['end']
                
                if start_time < (chunk_start_time_s + (CHUNK_SIZE_MS / 1000.0)):
                    srt_content += f"{caption_index}\n"
                    srt_content += f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n"
                    srt_content += f"{segment['text'].strip()}\n\n"
                    caption_index += 1

        except Exception as e:
            print(f"    ! Error transcribing chunk {i+1}: {e}")
    
    # --- 5. SAVING THE SRT FILE ---
    print(f"Step 5: Saving captions to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"\n✅ Caption generation complete for {chosen_file}!")
    print(f"   Your SRT file is saved at: {output_path}")

if __name__ == "__main__":
    main()