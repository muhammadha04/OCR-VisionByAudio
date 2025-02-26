# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "openai",
#     "opencv-python",
#     "pupil-labs-realtime-api",
#     "pydub",
#     "python-dotenv",
#     "pillow",  # For image handling in Tkinter
#     "tk",      # For GUI
# ]
# ///

import os
import tempfile
import base64
import io
from typing import Literal
import cv2
from dotenv import load_dotenv
from openai import OpenAI
from pupil_labs.realtime_api.simple import discover_one_device
from pydub import AudioSegment
from pydub.playback import play
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import scrolledtext

load_dotenv()

# Ensure the folder exists and is writable
os.makedirs(r"C:\Users\Muhammad\Downloads\wavs", exist_ok=True)
tempfile.tempdir = r"C:\Users\Muhammad\Downloads\wavs"

class OpenAICost:
    date_updated = "2024-10-21"
    "Cost $ per 1M token"
    model = {
        "gpt-4o": {
            "input_cost": 2.5,
            "output_cost": 10,
            "neon_frame_input": 0.001913,
        },
        "tts-1": {"output_cost": 15},
    }

    @classmethod
    def input_cost(cls, model):
        return cls.model.get(model, {}).get("input_cost")

    @classmethod
    def output_cost(cls, model):
        return cls.model.get(model, {}).get("output_cost")

    @classmethod
    def frame_cost(cls, model):
        return cls.model.get(model, {}).get("neon_frame_input")

class Assistant:
    def __init__(self):
        self.device = None
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy"
        self.audio_format: Literal["mp3", "opus", "aac", "flac", "pcm", "wav"] = "opus"
        self.session_cost = 0
        self.running = True
        self.setup_prompts()
        self.initialise_device()
        self.setup_gui()

    def setup_prompts(self):
        self.prompts = {
            "base": (
                "You are an assistant for individuals with visual impairments. "
                "I will send you an image; please detect any whiteboards, extract the text, "
                "and provide a concise summary. Reply in English."
            )
        }

    def initialise_device(self):
        print("Looking for the next best device...")
        self.device = discover_one_device(max_search_duration_seconds=10)
        if self.device is None:
            print("No device found.")
            raise SystemExit(-1)
        print(f"Connected to {self.device}.")

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Assistant GUI")

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.status_label = tk.Label(self.root, text="Status: Initializing...")
        self.status_label.pack()

        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=50, height=10)
        self.text_area.pack()

        self.read_button = tk.Button(self.root, text="Read Text", command=self.read_text)
        self.read_button.pack()

    def process_frame(self):
        self.matched = self.device.receive_matched_scene_and_eyes_video_frames_and_gaze()
        if not self.matched:
            print("No matched frames received.")
            return

        frame = self.matched.scene.bgr_pixels
        gaze_x, gaze_y = int(self.matched.gaze.x), int(self.matched.gaze.y)
        cv2.circle(frame, (gaze_x, gaze_y), radius=2, color=(0, 0, 255), thickness=1)

        self.display_frame(frame)
        self.status_label.config(text="Status: Processing frame...")

        self.encode_image(frame)
        self.assist()

    def display_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        self.video_label.imgtk = tk_image
        self.video_label.configure(image=tk_image)

    def encode_image(self, frame):
        _, buffer = cv2.imencode(".jpg", frame)
        self.base64Frame = base64.b64encode(buffer).decode("utf-8")

    def assist(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompts["base"]},
                {"role": "user", "content": "Here is the image.", "image": self.base64Frame},
            ],
            max_tokens=500,
        )

        response_text = response.choices[0].message.content
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, response_text)

        response_cost = (
            response.usage.prompt_tokens / 1e6 * OpenAICost.input_cost(self.model)
            + response.usage.completion_tokens / 1e6 * OpenAICost.output_cost(self.model)
            + OpenAICost.frame_cost(self.model)
        )
        self.session_cost += response_cost
        print(f"Response: {response_text}, Cost: ${response_cost:.6f}, Total: ${self.session_cost:.6f}")

    def read_text(self):
        text = self.text_area.get(1.0, tk.END).strip()
        if text:
            response_audio = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                response_format=self.audio_format,
                input=text,
            )
            byte_stream = io.BytesIO(response_audio.content)
            audio = AudioSegment.from_file(byte_stream)
            play(audio)
        else:
            print("No text to read.")

    def run(self):
        self.status_label.config(text="Status: Running...")
        self.root.after(0, self.process_frame)
        self.root.mainloop()
        self.device.close()
        print(f"Session ended. Total cost: ${self.session_cost:.6f}")

if __name__ == "__main__":
    assistant = Assistant()
    assistant.run()
