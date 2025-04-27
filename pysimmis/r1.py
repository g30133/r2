#!/usr/bin/env python3
"""
Simmis Assistant - A Multimodal Desktop AI Assistant

This Python script provides a simplified yet powerful desktop assistant that:
1. Captures screenshots of your screen to see what you're looking at
2. Records audio from your microphone when you speak
3. Transcribes your speech using OpenAI's Whisper model
4. Sends both the screenshot and transcription to OpenAI's ChatGPT-4o model
5. Processes the AI's response to generate spoken replies
6. Plays back the audio response through your system's speakers

Usage:
    python3 simmis_assistant.py [--api-key OPENAI_API_KEY] [--user USERNAME]

Requirements:
    - ffmpeg/ffplay for audio recording and playback
    - pyautogui for screenshots
    - openai Python package
    
The assistant maintains conversation history, allowing for context-aware 
interactions with your screen content.
"""
# Standard imports
import sys
import os
import json
import time
import base64
import tempfile
import uuid
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional

# External dependencies
import pyautogui  # For screenshots
from openai import OpenAI  # For OpenAI API access

import pyaudio
import wave

import audio_input
import audio_output

# Suppress unnecessary warnings
os.environ['NOISY_WARNINGS'] = '0'

# Default to None, will use environment variable
DEFAULT_OPENAI_KEY = None

# Configuration class for the assistant
class Config:
    def __init__(self, api_key: str = DEFAULT_OPENAI_KEY, user: str = "User"):
        self.api_key = api_key
        self.user = user
        self.audio_interval = 5  # seconds
        self.screen_interval = 10  # seconds

# Stores conversation history and events
class EventStore:
    def __init__(self, max_history: int = 100):
        self.events: List[Dict[str, Any]] = []
        self.max_history = max_history
    
    def add_event(self, event_type: str, role: str, **kwargs):
        """Add a new event and trim history if needed"""
        event = {
            "event/type": event_type,
            "event/created": datetime.now(),
            "event/role": role,
            **kwargs
        }
        self.events.append(event)
        
        # Trim history if exceeding max size
        if len(self.events) > self.max_history:
            self.events = self.events[-self.max_history:]
            
        return event
    
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the most recent events, sorted by creation time"""
        return sorted(self.events[-limit:], key=lambda x: x["event/created"])

# Handles all OpenAI API calls (chat, speech-to-text, text-to-speech)
class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def chat(self, messages: List[Dict[str, Any]]) -> str:
        """Call the OpenAI chat API using ChatGPT-4o model"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use the standard model name for better compatibility
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in OpenAI chat: {e}")
            return "I couldn't process that. Please try again."
    
    def text_to_speech(self, text: str) -> str:
        """Convert text to speech using OpenAI's TTS API"""
        if not text:
            return ""
            
        # Create a temporary file for the audio
        temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
        print(f"tts_temp_file: {temp_file}")
        #temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        #print(f"temp_file: {temp_file}")

        try:
            # Create speech with OpenAI's TTS API
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="alloy", 
                input=text
            )
            
            # Write audio data
            with open(temp_file, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
            
            return temp_file
                
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            return ""
    
    def speech_to_text(self, audio_file: str) -> str:
        """Convert speech to text using OpenAI's Whisper API"""
        try:
            with open(audio_file, "rb") as file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=file,
                    language="en"
                )
                return response.text
        except Exception as e:
            print(f"Error in speech-to-text: {e}")
            return ""

# Handles audio recording, playback, and screenshot capture
class InputHandler:
    def __init__(self):
        # Check if ffmpeg is available
        if os.system("which ffmpeg >/dev/null 2>&1") != 0:
            print("ERROR: ffmpeg not found. Please install it.")
            sys.exit(1)
    
    def listen_microphone(self, duration: int = 5, ctx_in = None) -> str:
        #"""Record audio from microphone using ffmpeg"""
        temp_file = os.path.join(tempfile.gettempdir(), f"microphone-{uuid.uuid4()}.wav")
        
        print(f"========= Recording audio for {duration} seconds... on {temp_file}")
        #cmd = f"ffmpeg -f alsa -i default -ar 16000 -ac 1 -t {duration} -y {temp_file} -loglevel quiet"
        #cmd = f"ffmpeg -f pulse -i AWS-Virtual-Microphone -ar 16000 -ac 1 -t {duration} -y {temp_file} -loglevel quiet"
        #cmd = f"ffmpeg -f pulse -i agent_input.monitor -ar 16000 -ac 1 -t {duration} -y {temp_file} -loglevel error"
        #exit_code = os.system(cmd)
        exit_code = audio_input.record(temp_file, duration, *ctx_in)
        if exit_code == 0 and os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
            return temp_file
        return ""

    def play_audio(self, file_path: str, ctx_out):
        print(f"play_audio(): {file_path}")
        """Play an audio file using ffplay"""
        if not os.path.exists(file_path):
            return
            
        try:
            #os.system(f'PULSE_SINK=auto_null ffplay -nodisp -autoexit -loglevel quiet "{file_path}"')
            #os.system(f'PULSE_SINK=agent_output ffplay -nodisp -autoexit -loglevel quiet "{file_path}"')
            #os.system(f'paplay --device=agent_output "{file_path}"')
            # convert mp3 to wav(48000 hz, 2 channel, 16le)
            os.system(f"ffmpeg -i {file_path} -ar 48000 -ac 2 /tmp/output.wav")
            audio_output.play("/tmp/output.wav", *ctx_out)
        finally:
            # Clean up the file
            try:
                os.remove(file_path)
                os.remove("/tmp/output.wav")
            except:
                pass
    
    def take_screenshot(self) -> str:
        """Take a screenshot and save to temp file"""
        temp_file = os.path.join(tempfile.gettempdir(), f"screenshot-{uuid.uuid4()}.png")
        screenshot = pyautogui.screenshot()
        screenshot.save(temp_file)
        return temp_file
    
    def encode_file(self, file_path: str) -> str:
        """Encode a file to base64"""
        with open(file_path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

# Parses and executes assistant actions (statements and thoughts)
class ActionHandler:
    def parse_actions(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON actions from the response"""
        default_actions = [
            {"action": "statement", "text": "I'm listening. How can I help you?"}
        ]
        
        if not response:
            return default_actions
        
        try:
            # Extract JSON if it's wrapped in markdown code blocks
            if "```json" in response:
                parts = response.split("```json")
                json_str = parts[1].split("```")[0].strip()
            elif "```" in response:
                parts = response.split("```")
                json_str = parts[1].strip()
            else:
                json_str = response.strip()
            
            # Parse JSON
            actions = json.loads(json_str)
            
            # Return actions
            if isinstance(actions, list):
                return actions
            elif isinstance(actions, dict):
                return [actions]
            else:
                return default_actions
                
        except:
            return default_actions
    
    def execute_action(self, action: Dict[str, Any], openai_client: OpenAIClient, input_handler: InputHandler, event_store=None, ctx_out=None):
        """Execute a single action"""
        action_type = action.get("action", "")
        
        if action_type == "statement":
            text = action.get("text", "")
            if text:
                print(f"Assistant: {text}")
                
                # Store audio output in the event store for history
                if event_store:
                    event_store.add_event(
                        "is.simm.runtimes.ubuntu/audio-out",
                        "assistant",
                        **{"audio/out": text}
                    )
                
                # Generate and play audio
                audio_file = openai_client.text_to_speech(text)
                if audio_file:
                    input_handler.play_audio(audio_file, ctx_out)
        
        elif action_type == "inner-monologue":
            text = action.get("text", "")
            print(f"[Inner thought]: {text}")
            
            # Store inner monologue in the event store for history
            if event_store:
                event_store.add_event(
                    "is.simm.runtimes.ubuntu/inner-monologue",
                    "assistant", 
                    **{"inner/thought": text}
                )

# Main assistant class that coordinates all components
class SimmisPythonAssistant:
    def __init__(self, config: Config):
        self.config = config
        self.event_store = EventStore()
        self.openai_client = OpenAIClient(config.api_key)
        self.input_handler = InputHandler()
        self.action_handler = ActionHandler()
        self.running = False
        
        # Simplified system prompt
        self.system_prompt = """
You are a supervisor actively controlling the Minecraft environment from STEVE1’s perspective. Your job is to generate short prompts to give to STEVE1 to execute next.
You see a “screenshot” of what STEVE1 sees and you can access the current inventory status. 
Keep track of previous actions, environment context, and inventory to maintain continuity. 

For example, if you notice trees and no wood in the inventory, command STEVE1 to gather wood by outptting the two words "gather wood".
Your only output should be the task to do.

The user’s name is {user}.

Instructions: 
- Maintain continuity by considering previous conversation history and environment status.
- Respond to direct user questions as needed.
- Generate short, low-level commands to pilot STEVE1 (e.g., “gather wood”, "dig dirt", "swim in water", “place torch,” etc.), rather than high-level tasks
- Using understanding of current situation, give higher level tasks such as "build a crafting table", "build a wooden pickaxe", and so on.
- the generated prompts must be very basic.

Your output must always be a **markdown JSON block** using the following JSON format:

json
[
  {{"action": "statement", "text": "I can see you're looking at a document about climate change. The graph shows temperature increases over the past century."}},
  {{"action": "inner-monologue", "text": "The user is looking at a scientific paper with charts about global warming."}}
]

Always respond with a markdown JSON block of actions.
"""

    def events_to_messages(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert events to OpenAI message format - includes all conversation history"""
        messages = []
        
        for event in events:
            event_type = event.get("event/type", "")
            role = event.get("event/role", "user")
            
            if event_type == "is.simm.runtimes.ubuntu/audio-in":
                # User speech input
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "text", 
                        "text": f"User said: {event.get('audio/in', '')}"
                    }]
                })
            
            elif event_type == "is.simm.runtimes.ubuntu/audio-out":
                # Assistant speech output
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": f"Assistant said: {event.get('audio/out', '')}"
                    }]
                })
                
            elif event_type == "is.simm.runtimes.ubuntu/inner-monologue":
                # Include inner thoughts explicitly
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": f"Assistant thought: {event.get('inner/thought', '')}"
                    }]
                })
                
            elif event_type == "is.simm.runtimes.ubuntu/assistant-output":
                # The full JSON response from the assistant
                # We include a brief summary to maintain context between interactions
                assistant_output = event.get("assistant/output", "")
                if len(assistant_output) > 100:  # Only if it's substantial
                    messages.append({
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": f"Assistant's previous raw response: {assistant_output[:100]}..."
                        }]
                    })
            
        return messages
    
    def run(self):
        """Run the assistant in a continuous loop"""
        self.running = True
        print(f"Starting Simmis Assistant for user: {self.config.user}")
        print("Listening for your questions... (Press Ctrl+C to stop)")
        
        try:
            ctx_in = audio_input.open()
            ctx_out = audio_output.open()

            while self.running:
                # 1. Record audio
                audio_file = self.input_handler.listen_microphone(self.config.audio_interval, ctx_in)
                if not audio_file:
                    time.sleep(1)
                    continue
                
                # 2. Transcribe audio
                text = self.openai_client.speech_to_text(audio_file)
                os.remove(audio_file)  # Clean up
                
                if not text:
                    continue
                
                # 3. Store user input
                print(f"User: {text}")
                self.event_store.add_event(
                    "is.simm.runtimes.ubuntu/audio-in",
                    "user",
                    **{"audio/in": text}
                )
                
                # 4. Take screenshot
                print("take screenshot >>>")
                screenshot_path = self.input_handler.take_screenshot()
                print("<<< take screenshot")
                
                # 5. Prepare messages for OpenAI
                print("prep msg for openai >>>")
                events = self.event_store.get_recent_events()
                messages = self.events_to_messages(events)
                print(f"Including {len(messages)} messages from conversation history")
                
                system_message = {
                    "role": "system",
                    "content": self.system_prompt.format(user=self.config.user)
                }
                print("before all_messages")
                all_messages = [system_message] + messages
                print("add screenshot >>>")
                
                # 6. Add screenshot
                screenshot_base64 = self.input_handler.encode_file(screenshot_path)
                all_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Current screenshot:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            }
                        }
                    ]
                })

                print("<< add screenshot")
                # 7. Call OpenAI API
                response = self.openai_client.chat(all_messages)
                self.event_store.add_event(
                    "is.simm.runtimes.ubuntu/assistant-output",
                    "assistant",
                    **{"assistant/output": response}
                )
                
                # 8. Process response
                actions = self.action_handler.parse_actions(response)
                for action in actions:
                    self.action_handler.execute_action(action, self.openai_client, self.input_handler, self.event_store, ctx_out)
                
                # 9. Clean up
                os.remove(screenshot_path)
                
                # 10. Brief pause before next cycle
                #time.sleep(0.5)
                time.sleep(0.5)
        except Exception as e:
            print('e:', e)
            self.running = False
            audio_input.close()
            audio_output.close()
        '''
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self.running = False
            print("Assistant stopped")
        '''
    def stop(self):
        """Stop the assistant"""
        self.running = False

# Main function
# Nothing needed here - removing the test_interaction function

def main():
    """Main function to run the Simmis Assistant"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simmis Python Assistant")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (default: use from environment or config)")
    parser.add_argument("--user", type=str, default="User", help="User name")
    args = parser.parse_args()
    
    # Redirect stderr to suppress audio errors
    sys.stderr = open(os.devnull, 'w')
    
    # Get API key with priority: 1) command line 2) environment
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or DEFAULT_OPENAI_KEY
    
    if not api_key:
        print("Error: OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    # Create and run assistant
    config = Config(api_key=api_key, user=args.user)
    assistant = SimmisPythonAssistant(config)
    
    print(f"Starting Simmis Assistant for {args.user}")
    print("Press Ctrl+C to exit")
    print("-----------------------------------")
    
    try:
        # Run normally
        assistant.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
        assistant.stop()

if __name__ == "__main__":
    main()
