# Simplified Simmis Assistant

A simplified multimodal AI desktop assistant that can see your screen and hear your voice.

## Features

- **Screen Vision**: Captures screenshots to analyze what you're viewing
- **Voice Recognition**: Records audio from your microphone
- **Speech-to-Text**: Transcribes your speech using OpenAI's Whisper
- **Natural Interaction**: Processes multimodal inputs with OpenAI's GPT-4o
- **Audio Replies**: Converts responses to speech using OpenAI's TTS
- **Continuous Context**: Maintains conversation history for context-aware responses

## Requirements

- Python 3.9+
- ffmpeg (for audio recording and playback)
- OpenAI API Key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/plai-group/pysimmis.git
   cd pysimmis
   ```

2. Create a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install ffmpeg if not already available:
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

1. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY="your-api-key"
   ```

2. Run the assistant:
   ```
   python simmis_assistant.py
   ```

3. Additional options:
   ```
   python simmis_assistant.py --user YourName  # Customize the user name
   python simmis_assistant.py --api-key "your-api-key"  # Provide API key via command line
   ```

4. Speak to the assistant and allow it to see your screen. Press Ctrl+C to exit.

## How It Works

1. The assistant records audio from your microphone
2. Audio is transcribed to text using OpenAI's Whisper API
3. A screenshot is captured and encoded
4. Both the transcription and image are sent to OpenAI's GPT-4o
5. The assistant processes the response and generates spoken replies
6. The response is played through your speakers

## License

[MIT License](LICENSE)
