import subprocess
import sys

def main():
    print("=== Video Transcriptor ===\n")
    print("Select transcription mode:")
    print("  [1] Online  (Groq API - fast, requires API key)")
    print("  [2] Local   (Whisper - runs on your machine, requires GPU recommended)")
    print()

    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == "1":
                subprocess.run([sys.executable, "main_online.py"])
                break
            elif choice == "2":
                subprocess.run([sys.executable, "main_local.py"])
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled by user.")
            return

if __name__ == "__main__":
    main()
