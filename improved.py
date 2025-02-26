import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import datetime
import uuid
import time

# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "openai",
#     "opencv-python",
#     "pupil-labs-realtime-api",
#     "pydub",
#     "python-dotenv",
# ]
# ///

import base64
import io
import os
from typing import Literal

import cv2
from dotenv import load_dotenv
from openai import OpenAI
from pupil_labs.realtime_api.simple import discover_one_device
from pydub import AudioSegment
from pydub.playback import play

class NotebookEntry:
    def __init__(self, image, title, description, timestamp, entry_id=None):
        self.image = image
        self.title = title
        self.description = description
        self.timestamp = timestamp
        self.entry_id = entry_id or str(uuid.uuid4())[:8]

class ImprovedAssistant():
    def __init__(self):
        super().__init__()
        self.setup_gui()
        self.entries = []
        
    def setup_gui(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Assistive Technology Interface")
        self.root.geometry("1200x800")
        
        # Create left panel for camera feed
        self.camera_frame = ttk.Frame(self.root, width=400, height=300)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")
        
        # Create camera label
        self.camera_label = ttk.Label(self.camera_frame)
        self.camera_label.grid(row=0, column=0)
        
        # Create right panel for notebook
        self.notebook_frame = ttk.Frame(self.root)
        self.notebook_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Create canvas and scrollbar for notebook
        self.canvas = tk.Canvas(self.notebook_frame)
        self.scrollbar = ttk.Scrollbar(self.notebook_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Grid layout for notebook
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        
        # Configure grid weights
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.notebook_frame.grid_columnconfigure(0, weight=1)
        self.notebook_frame.grid_rowconfigure(0, weight=1)
        
        # Mode indicator
        self.mode_label = ttk.Label(self.camera_frame, text="Current Mode: describe")
        self.mode_label.grid(row=1, column=0, pady=5)
        
    def update_camera_feed(self):
        if hasattr(self, 'bgr_pixels'):
            # Resize frame for display
            display_size = (380, 285)
            frame = cv2.resize(self.bgr_pixels, display_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update label
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo
            
            # Update mode indicator
            self.mode_label.configure(text=f"Current Mode: {self.mode}")
        
        # Schedule next update
        self.root.after(10, self.update_camera_feed)
        
    def add_notebook_entry(self, image, gpt_response):
        # Generate title from GPT (you'll need to modify the prompt)
        title_prompt = "Please provide a brief title (max 5 words) for this image and scene:"
        title_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": title_prompt},
                {"role": "user", "content": [{"image": self.base64Frame, "resize": 768}]},
            ],
            max_tokens=20,
        )
        title = title_response.choices[0].message.content
        
        # Create new entry
        timestamp = datetime.datetime.now()
        entry = NotebookEntry(image, title, gpt_response, timestamp)
        self.entries.append(entry)
        
        # Create frame for entry
        entry_frame = ttk.Frame(self.scrollable_frame)
        entry_frame.grid(row=len(self.entries)-1, column=0, pady=10, padx=10, sticky="ew")
        
        # Add title
        title_label = ttk.Label(
            entry_frame, 
            text=f"[{entry.entry_id}] {entry.title}",
            font=('Helvetica', 12, 'bold')
        )
        title_label.grid(row=0, column=0, sticky="w")
        
        # Add timestamp
        time_label = ttk.Label(
            entry_frame,
            text=entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            font=('Helvetica', 8)
        )
        time_label.grid(row=1, column=0, sticky="w")
        
        # Add description
        desc_label = ttk.Label(
            entry_frame,
            text=entry.description,
            wraplength=600
        )
        desc_label.grid(row=2, column=0, sticky="w")
        
        # Add separator
        ttk.Separator(self.scrollable_frame, orient='horizontal').grid(
            row=len(self.entries)*2-1, column=0, sticky="ew", pady=5
        )
        
    def process_frame(self):
        self.matched = self.device.receive_matched_scene_and_eyes_video_frames_and_gaze()
        if not self.matched:
            print("Not able to find a match!")
            return
            
        # Draw gaze circle
        cv2.circle(
            self.matched.scene.bgr_pixels,
            (int(self.matched.gaze.x), int(self.matched.gaze.y)),
            radius=2,
            color=(0, 0, 255),
            thickness=1,
        )
        self.bgr_pixels = self.matched.scene.bgr_pixels
        
        # Handle key events
        key = cv2.waitKey(1) & 0xFF
        if key in self.key_actions:
            self.key_actions[key]()
            
    def handle_space(self):
        self.encode_image()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompts["base"] + self.prompts[self.mode],
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        "Here goes the image",
                        {"image": self.base64Frame, "resize": 768},
                    ],
                },
            ],
            max_tokens=200,
        )
        
        # Add entry to notebook
        self.add_notebook_entry(self.bgr_pixels.copy(), response.choices[0].message.content)
        
        # Create and play audio response
        self.create_and_play_audio(response.choices[0].message.content)
        
    def create_and_play_audio(self, text):
        response_audio = self.client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            response_format=self.audio_format,
            input=text,
        )
        byte_stream = io.BytesIO(response_audio.content)
        audio = AudioSegment.from_file(byte_stream)
        audio = audio.speedup(playback_speed=1.1)
        play(audio)
        
    def run(self):
        self.update_camera_feed()
        while self.device is not None and self.running:
            self.process_frame()
            self.root.update()
        
        print("Stopping...")
        print(f"Total session cost {self.session_cost}$")
        self.device.close()
        self.root.destroy()

load_dotenv()

class OpenAICost:
    date_updated = "2024-10-21"
    "Cost $ per 1M token"
    model = {
        "gpt-4o-audio-preview": {
            "input_cost": 2.5,
            "input_cost_cached": 1.25,
            "output_cost": 10,
            "output_audio_cost": 200,
            "neon_frame_input": 0.001913,
            # Currently does not accept image input
        },
        "gpt-4o": {
            "input_cost": 2.5,
            "input_cost_cached": 1.25,
            "output_cost": 10,
            "neon_frame_input": 0.001913,
        },
        "tts-1": {"output_cost": 15},
        "tts-1-hd": {"output_cost": 30},
    }

    @classmethod
    def input_cost(cls, model):
        if model in cls.model and "input_cost" in cls.model[model]:
            return cls.model[model]["input_cost"]
        else:
            return None

    @classmethod
    def output_cost(cls, model):
        if model in cls.model and "output_cost" in cls.model[model]:
            return cls.model[model]["output_cost"]
        else:
            return None

    @classmethod
    def frame_cost(cls, model):
        if model in cls.model and "neon_frame_input" in cls.model[model]:
            return cls.model[model]["neon_frame_input"]
        else:
            return None


class Assistant:
    def __init__(self):
        self.device = None
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"  # "gpt-4-turbo"
        self.setup_prompts()
        self.mode = "describe"
        self.running = True
        self.key_actions = {
            ord("a"): lambda: setattr(self, "mode", "describe"),
            ord("s"): lambda: setattr(self, "mode", "dangers"),
            ord("d"): lambda: setattr(self, "mode", "intention"),
            ord("f"): lambda: setattr(self, "mode", "in_detail"),
            32: self.handle_space,
            27: lambda: setattr(self, "running", False),
        }
        self.session_cost = 0
        self.voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = (
            "alloy"
        )
        self.audio_format: Literal["mp3", "opus", "aac", "flac", "pcm", "wav"] = "opus"
        self.initialise_device()

    def initialise_device(self):
        print("Looking for the next best device...")
        self.device = discover_one_device(max_search_duration_seconds=10)
        if self.device is None:
            print("No device found.")
            raise SystemExit(-1)

        print(f"Connecting to {self.device}...")

    def setup_prompts(self):
        self.prompts = {
            "base": "You are a visual and communication aid for individuals with visual impairment"
            + "(low vision) or communication difficulties, they are wearing eye-tracking glasses, "
            + "I am sending you an image with a red circle indicating the wearer's gaze, do not "
            + "describe the whole image unless explicitly asked, be succinct, reply in: English",
            "describe": "in couple of words (max. 8) say what the person is looking at.",
            "dangers": "briefly indicate if there is any posing risk for the person in the scene, "
            + "be succinct (max 30 words).",
            "intention": "given that the wearer has mobility and speaking difficulties, "
            + "briefly try to infer the wearer's intention based on what they are looking "
            + "at (maximum of 30 words).",
            "in_detail": "describe the scene in detail, with a maximum duration of one minute of speaking.",
        }

    def process_frame(self):
        self.matched = (
            self.device.receive_matched_scene_and_eyes_video_frames_and_gaze()
        )
        if not self.matched:
            print("Not able to find a match!")
            return
        self.annotate_and_show_frame()

    def annotate_and_show_frame(self):
        cv2.circle(
            self.matched.scene.bgr_pixels,
            (int(self.matched.gaze.x), int(self.matched.gaze.y)),
            radius=2,
            color=(0, 0, 255),
            thickness=1,
        )
        self.bgr_pixels = self.matched.scene.bgr_pixels
        self.bgr_pixels = cv2.putText(
            self.bgr_pixels,
            str(self.mode),
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2,
            cv2.LINE_8,
        )
        cv2.imshow(
            "Scene camera with eyes and gaze overlay",
            self.bgr_pixels,
        )
        key = cv2.waitKey(1) & 0xFF
        if key in self.key_actions:
            self.key_actions[key]()

    def encode_image(self):
        start_time = time.time()
        _, buffer = cv2.imencode(".jpg", self.matched.scene.bgr_pixels)
        self.base64Frame = base64.b64encode(buffer).decode("utf-8")
        print(f"Image encoding time: {time.time() - start_time:.4f} seconds")

    def assist(self):
        # GPT processing time
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompts["base"] + self.prompts[self.mode],
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        "Here goes the image",
                        {"image": self.base64Frame, "resize": 768},
                    ],
                },
            ],
            max_tokens=200,
        )
        print(f"GPT response time: {time.time() - start_time:.4f} seconds")

        response_cost = (
            response.usage.prompt_tokens / int(1e6) * OpenAICost.input_cost(self.model)
            + response.usage.completion_tokens
            / int(1e6)
            * OpenAICost.output_cost(self.model)
            + OpenAICost.frame_cost(self.model)
        )

        # TTS processing time
        start_time = time.time()
        response_audio = self.client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            response_format=self.audio_format,
            input=response.choices[0].message.content,
        )
        print(f"TTS response time: {time.time() - start_time:.4f} seconds")

        TTS_cost = (
            len(response.choices[0].message.content)
            / int(1e6)
            * OpenAICost.output_cost("tts-1")
        )
        byte_stream = io.BytesIO(response_audio.content)
        audio = AudioSegment.from_file(byte_stream)
        self.session_cost += response_cost + TTS_cost
        print(
            f"R: {response.choices[0].message.content}, approx cost(GPT/TTS): {response_cost} / {TTS_cost} $ Total: {response_cost+TTS_cost} $"
        )

        # Audio playback
        start_time = time.time()
        audio = audio.speedup(playback_speed=1.1)
        play(audio)
        print(f"Audio playback time: {time.time() - start_time:.4f} seconds")

    def handle_space(self):
        self.encode_image()
        self.assist()

    def run(self):
        while self.device is not None and self.running:
            self.process_frame()
        print("Stopping...")
        print(f"Total session cost {self.session_cost}$")
        self.device.close()  # explicitly stop auto-update


if __name__ == "__main__":
    eyes = ImprovedAssistant()
    eyes.run()