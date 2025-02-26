def quit(self, event=None):
        self.running = False
import time
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
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import datetime
import uuid

load_dotenv()
import csv
import os
from datetime import datetime  # Change this line

class PerformanceLogger:
    def __init__(self, filename="performance_logs.csv"):
        self.filename = filename
        self.initialize_csv()
    
    def initialize_csv(self):
        # Create file with headers if it doesn't exist
        if not os.path.exists(self.filename):
            headers = [
                'timestamp',
                'image_encoding_time',
                'gpt_response_time',
                'tts_response_time',
                'audio_playback_time',
                'total_response_time',
                'gpt_tokens_prompt',
                'gpt_tokens_completion',
                'gpt_cost',
                'tts_cost',
                'total_cost',
                'response_text'
            ]
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_performance(self, metrics):
        try:
            with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics['timestamp'],
                    f"{metrics['image_encoding_time']:.4f}",
                    f"{metrics['gpt_response_time']:.4f}",
                    f"{metrics['tts_response_time']:.4f}",
                    f"{metrics['audio_playback_time']:.4f}",
                    f"{metrics['total_response_time']:.4f}",
                    metrics['gpt_tokens_prompt'],
                    metrics['gpt_tokens_completion'],
                    f"{metrics['gpt_cost']:.6f}",
                    f"{metrics['tts_cost']:.6f}",
                    f"{metrics['total_cost']:.6f}",
                    metrics['response_text']
                ])
        except Exception as e:
            print(f"Error logging performance metrics: {e}")

# Keep your original OpenAICost class exactly as is
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

class NotebookEntry:
    def __init__(self, description, timestamp):
        self.description = description
        self.timestamp = timestamp
        self.entry_id = str(uuid.uuid4())[:8]

class Assistant:
    def __init__(self):
        # Keep all your original initializations
        self.muted = False
        self.device = None
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.setup_prompts()
        self.mode = "describe"
        self.running = True
        self.logger = PerformanceLogger()

        self.key_actions = {
            # ord("a"): lambda: setattr(self, "mode", "describe"),
            # ord("s"): lambda: setattr(self, "mode", "dangers"),
            # ord("d"): lambda: setattr(self, "mode", "intention"),
            # ord("f"): lambda: setattr(self, "mode", "in_detail"),
            # 32: self.handle_space,
            # 27: lambda: setattr(self, "running", False),
        }
        self.session_cost = 0
        self.voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy"
        self.audio_format: Literal["mp3", "opus", "aac", "flac", "pcm", "wav"] = "opus"
        
        # Add GUI initialization
        self.setup_gui()
        self.entries = []
        
        # Initialize device last, just like in original
        self.initialise_device()
    def toggle_mute(self):
        self.muted = not self.muted
        # Update button text
        new_text = "🔇 Unmute" if self.muted else "🔊 Mute"
        self.mute_button.configure(text=new_text)

    def setup_gui(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Eye Tracking Assistant")
        self.root.geometry("1600x900")
        
        # Create main horizontal PanedWindow
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Notebook
        notebook_frame = ttk.Frame(self.main_paned)
        
        # Add formatting toolbar
        toolbar_frame = ttk.Frame(notebook_frame)
        toolbar_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Font size dropdown
        self.font_sizes = ['8', '10', '12', '14', '16', '18', '20', '24', '28', '32']
        self.font_size_var = tk.StringVar(value='12')
        font_size_menu = ttk.Combobox(toolbar_frame, textvariable=self.font_size_var,
                                     values=self.font_sizes, width=3)
        font_size_menu.pack(side=tk.LEFT, padx=2)
        font_size_menu.bind('<<ComboboxSelected>>', self.change_font_size)
        
        # Font family dropdown
        self.font_families = ['Arial', 'Verdana', 'Tahoma', 'Comic Sans MS', 'OpenDyslexic']
        self.font_family_var = tk.StringVar(value='Arial')
        font_family_menu = ttk.Combobox(toolbar_frame, textvariable=self.font_family_var,
                                       values=self.font_families, width=15)
        font_family_menu.pack(side=tk.LEFT, padx=2)
        font_family_menu.bind('<<ComboboxSelected>>', self.change_font_family)
        
        # Text color button
        color_btn = ttk.Button(toolbar_frame, text="Text Color", command=self.choose_color)
        color_btn.pack(side=tk.LEFT, padx=2)
        
        # Background color button
        bg_color_btn = ttk.Button(toolbar_frame, text="Background", command=self.choose_bg_color)
        bg_color_btn.pack(side=tk.LEFT, padx=2)
        
        # Line spacing dropdown
        self.line_spacings = ['1.0', '1.5', '2.0', '2.5', '3.0']
        self.line_spacing_var = tk.StringVar(value='1.0')
        line_spacing_menu = ttk.Combobox(toolbar_frame, textvariable=self.line_spacing_var,
                                        values=self.line_spacings, width=3)
        line_spacing_menu.pack(side=tk.LEFT, padx=2)
        line_spacing_menu.bind('<<ComboboxSelected>>', self.change_line_spacing)
        
        # Create tags for formatting
        self.text_tags = {}
        
        # Notebook scrollable text area
        self.text_area = tk.Text(notebook_frame, wrap=tk.WORD, font=('Arial', 12))
        notebook_scroll = ttk.Scrollbar(notebook_frame, command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=notebook_scroll.set)
        
        notebook_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right side setup
        right_container = ttk.Frame(self.main_paned)
        right_paned = ttk.PanedWindow(right_container, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)
        
        # Top right - Camera feed and control buttons
        self.video_frame = ttk.Frame(right_paned)
        # Add mute toggle button in its own frame
        mute_frame = ttk.Frame(self.video_frame)
        mute_frame.pack(fill=tk.X, padx=5, pady=(5,0))
        
        self.mute_button = ttk.Button(mute_frame, 
                                    text="🔊 Mute",
                                    command=self.toggle_mute,
                                    style='Mute.TButton')
        self.mute_button.pack(fill=tk.X, padx=2)
        # Add control buttons above camera feed
        button_frame = ttk.Frame(self.video_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Style for the buttons
        style = ttk.Style()
        style.configure('Mode.TButton', padding=5)
        
        # # Mode buttons
        # describe_btn = ttk.Button(button_frame, text="Describe (A)", 
        #                         command=lambda: setattr(self, "mode", "describe"),
        #                         style='Mode.TButton')
        # describe_btn.pack(side=tk.LEFT, padx=2, expand=True)
        
        # dangers_btn = ttk.Button(button_frame, text="Dangers (S)", 
        #                        command=lambda: setattr(self, "mode", "dangers"),
        #                        style='Mode.TButton')
        # dangers_btn.pack(side=tk.LEFT, padx=2, expand=True)
        
        # intention_btn = ttk.Button(button_frame, text="Intention (D)", 
        #                          command=lambda: setattr(self, "mode", "intention"),
        #                          style='Mode.TButton')
        # intention_btn.pack(side=tk.LEFT, padx=2, expand=True)
        
        # detail_btn = ttk.Button(button_frame, text="In Detail (F)", 
        #                       command=lambda: setattr(self, "mode", "in_detail"),
        #                       style='Mode.TButton')
        # detail_btn.pack(side=tk.LEFT, padx=2, expand=True)
        
        capture_btn = ttk.Button(button_frame, text="Capture (Space)", 
                               command=self.handle_space,
                               style='Mode.TButton')
        capture_btn.pack(side=tk.LEFT, padx=2, expand=True)
        
        # Camera feed label
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom right - Console output
        console_frame = ttk.Frame(right_paned)
        
        self.console = tk.Text(console_frame, wrap=tk.WORD, 
                             bg='black', fg='white', 
                             font=('Consolas', 9))
        console_scroll = ttk.Scrollbar(console_frame, command=self.console.yview)
        self.console.configure(yscrollcommand=console_scroll.set)
        
        console_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.console.configure(state='disabled')
        
        # Add frames to the PanedWindows
        self.main_paned.add(notebook_frame, weight=3)
        self.main_paned.add(right_container, weight=1)
        
        right_paned.add(self.video_frame, weight=1)
        right_paned.add(console_frame, weight=1)
        
        # Keep keyboard shortcuts as fallback
        # self.root.bind('<a>', lambda e: setattr(self, "mode", "describe"))
        # self.root.bind('<s>', lambda e: setattr(self, "mode", "dangers"))
        # self.root.bind('<d>', lambda e: setattr(self, "mode", "intention"))
        # self.root.bind('<f>', lambda e: setattr(self, "mode", "in_detail"))
        # self.root.bind('<space>', lambda e: self.handle_space())
        
        # Redirect stdout to console
        import sys
        class ConsoleRedirector:
            def __init__(self, widget):
                self.widget = widget
            
            def write(self, text):
                self.widget.configure(state='normal')
                self.widget.insert(tk.END, text)
                self.widget.see(tk.END)
                self.widget.configure(state='disabled')
            
            def flush(self):
                pass
        
        sys.stdout = ConsoleRedirector(self.console)
        
        # Focus on root window to capture key events
        self.root.focus_set()

    # Keep your original prompts setup exactly as is
    def setup_prompts(self):
        self.prompts = {
            "base": "You are a visual and communication aid for individuals with visual impairment"
            + "(low vision) or communication difficulties, they are wearing eye-tracking glasses, "
            + "I am sending you an image with a red circle indicating the wearer's gaze, do not "
            + "describe the whole image unless explicitly asked, be succinct, reply in: English",
            "describe": "You are in a classroom with a whiteboard, detect the whiteboard, perform OCR on the text written on it",
            "dangers": "briefly indicate if there is any posing risk for the person in the scene, "
            + "be succinct (max 30 words).",
            "intention": "given that the wearer has mobility and speaking difficulties, "
            + "briefly try to infer the wearer's intention based on what they are looking "
            + "at (maximum of 30 words).",
            "in_detail": "describe the scene in detail, with a maximum duration of one minute of speaking.",
        }

    def initialise_device(self):
        print("Looking for the next best device...")
        self.device = discover_one_device(max_search_duration_seconds=10)
        if self.device is None:
            print("No device found.")
            raise SystemExit(-1)
        print(f"Connecting to {self.device}...")

    def process_frame(self):
        self.matched = self.device.receive_matched_scene_and_eyes_video_frames_and_gaze()
        if not self.matched:
            print("Not able to find a match!")
            return
            
        # Draw gaze circle and mode text exactly as in original
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
        
        # Convert for tkinter display - smaller size for right panel
        display_size = (400, 300)  # Adjusted size for right panel
        frame = cv2.resize(self.bgr_pixels, display_size)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=img_tk)
        self.video_label.image = img_tk

    def encode_image(self):
        start_time = time.time()
        _, buffer = cv2.imencode(".jpg", self.matched.scene.bgr_pixels)
        self.base64Frame = base64.b64encode(buffer).decode("utf-8")
        encoding_time = time.time() - start_time
        print(f"Image encoding time: {encoding_time:.4f} seconds")
        return encoding_time

    
    def assist(self):
        metrics = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'image_encoding_time': 0,
            'gpt_response_time': 0,
            'tts_response_time': 0,
            'audio_playback_time': 0,
            'total_response_time': 0,
            'gpt_tokens_prompt': 0,
            'gpt_tokens_completion': 0,
            'gpt_cost': 0,
            'tts_cost': 0,
            'total_cost': 0,
            'response_text': ''
        }

        total_start_time = time.time()

        # GPT processing
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
        metrics['gpt_response_time'] = time.time() - start_time
        print(f"GPT response time: {metrics['gpt_response_time']:.4f} seconds")

        # Calculate GPT costs
        gpt_cost = (
            response.usage.prompt_tokens / int(1e6) * OpenAICost.input_cost(self.model)
            + response.usage.completion_tokens
            / int(1e6)
            * OpenAICost.output_cost(self.model)
            + OpenAICost.frame_cost(self.model)
        )
        
        metrics['gpt_tokens_prompt'] = response.usage.prompt_tokens
        metrics['gpt_tokens_completion'] = response.usage.completion_tokens
        metrics['gpt_cost'] = gpt_cost
        metrics['response_text'] = response.choices[0].message.content

        # Add entry to notebook
        self.add_notebook_entry(response.choices[0].message.content)

        # TTS processing
        if not self.muted:
            start_time = time.time()
            response_audio = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                response_format=self.audio_format,
                input=response.choices[0].message.content,
            )
            metrics['tts_response_time'] = time.time() - start_time
            print(f"TTS response time: {metrics['tts_response_time']:.4f} seconds")

            metrics['tts_cost'] = (
                len(response.choices[0].message.content)
                / int(1e6)
                * OpenAICost.output_cost("tts-1")
            )
            
            # Audio playback
            byte_stream = io.BytesIO(response_audio.content)
            audio = AudioSegment.from_file(byte_stream)
            start_time = time.time()
            audio = audio.speedup(playback_speed=1.1)
            play(audio)
            metrics['audio_playback_time'] = time.time() - start_time
            print(f"Audio playback time: {metrics['audio_playback_time']:.4f} seconds")

        metrics['total_cost'] = metrics['gpt_cost'] + metrics['tts_cost']
        metrics['total_response_time'] = time.time() - total_start_time
        
        # Log metrics
        self.logger.log_performance(metrics)
        
        # Update session cost
        self.session_cost += metrics['total_cost']
        
        print(
            f"R: {metrics['response_text']}, approx cost(GPT/TTS): {metrics['gpt_cost']} / {metrics['tts_cost']} $ "
            f"Total: {metrics['total_cost']} $"
        )
    def choose_color(self):
        from tkinter import colorchooser
        color = colorchooser.askcolor(title="Choose Text Color")[1]
        if color:
            # Create a unique tag for this color
            tag_name = f"color_{color.replace('#', '')}"
            self.text_area.tag_configure(tag_name, foreground=color)
            # Apply to selected text
            try:
                sel_start = self.text_area.index("sel.first")
                sel_end = self.text_area.index("sel.last")
                self.text_area.tag_add(tag_name, sel_start, sel_end)
            except tk.TclError:  # No selection
                pass
    
    def choose_bg_color(self):
        from tkinter import colorchooser
        color = colorchooser.askcolor(title="Choose Background Color")[1]
        if color:
            # Create a unique tag for this background color
            tag_name = f"bg_{color.replace('#', '')}"
            self.text_area.tag_configure(tag_name, background=color)
            try:
                sel_start = self.text_area.index("sel.first")
                sel_end = self.text_area.index("sel.last")
                self.text_area.tag_add(tag_name, sel_start, sel_end)
            except tk.TclError:  # No selection
                pass
    
    def change_font_size(self, event=None):
        size = self.font_size_var.get()
        tag_name = f"size_{size}"
        self.text_area.tag_configure(tag_name, font=(self.font_family_var.get(), int(size)))
        try:
            sel_start = self.text_area.index("sel.first")
            sel_end = self.text_area.index("sel.last")
            self.text_area.tag_add(tag_name, sel_start, sel_end)
        except tk.TclError:  # No selection
            pass
    
    def change_font_family(self, event=None):
        family = self.font_family_var.get()
        tag_name = f"font_{family}"
        self.text_area.tag_configure(tag_name, font=(family, int(self.font_size_var.get())))
        try:
            sel_start = self.text_area.index("sel.first")
            sel_end = self.text_area.index("sel.last")
            self.text_area.tag_add(tag_name, sel_start, sel_end)
        except tk.TclError:  # No selection
            pass
    
    def change_line_spacing(self, event=None):
        spacing = float(self.line_spacing_var.get())
        self.text_area.configure(spacing3=spacing * 5)  # spacing3 affects between paragraphs
        
    def add_notebook_entry(self, text):
        # Insert entry with timestamp
        self.text_area.configure(state='normal')
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"\n[{timestamp}] {text}\n"
        self.text_area.insert(tk.END, entry)
        self.text_area.see(tk.END)
        self.text_area.configure(state='normal')  # Keep it editable    def quit(self, event=None):
    def handle_space(self):
        # Keep exactly as in original
        self.encode_image()
        self.assist()

    def run(self):
        while self.device is not None and self.running:
            self.process_frame()
            self.root.update()
        print("Stopping...")
        print(f"Total session cost {self.session_cost}$")
if __name__ == "__main__":
    eyes = Assistant()
    eyes.run()