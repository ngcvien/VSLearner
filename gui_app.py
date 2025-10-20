#!/usr/bin/env python3
# gui_app.py - Modern fullscreen GUI for sign language recognition
# Optimized for Raspberry Pi 4 with professional design

import os
import time
import json
import threading
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# mediapipe
import mediapipe as mp

# TFLite / Keras loading
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_AVAILABLE = True
except Exception:
    TFLITE_AVAILABLE = False
    try:
        import tensorflow as tf
        Interpreter = None
    except Exception:
        tf = None

# TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# Optional scaler
try:
    import pickle
    SCALER_AVAILABLE = True
except Exception:
    SCALER_AVAILABLE = False

# --------- CONFIG ----------
MODEL_TFLITE = "models/sign_model.tflite"
MODEL_H5 = "models/sign_model.h5"
LABELS_JSON = "models/labels.json"
SCALER_PKL = "models/scaler.pkl"

ROI_RATIO = 0.6
MIN_IN_FRAMES = 3
STABLE_FRAMES = 4
EXIT_TIMEOUT_MS = 450

# Color scheme (modern dark theme)
BG_COLOR = "#1a1a2e"
SECONDARY_BG = "#16213e"
ACCENT_COLOR = "#0f3460"
PRIMARY_COLOR = "#00d4ff"
TEXT_COLOR = "#eaeaea"
SUCCESS_COLOR = "#00ff88"
WARNING_COLOR = "#ff6b6b"
# --------------------------------

class ModernButton(tk.Canvas):
    """Custom modern button with hover effects"""
    def __init__(self, parent, text, command, width=200, height=50, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        bg=BG_COLOR, highlightthickness=0, **kwargs)
        
        self.command = command
        self.text = text
        self.width = width
        self.height = height
        self.hovered = False
        
        # Draw button
        self.rect = self.create_rectangle(2, 2, width-2, height-2,
                                         fill=ACCENT_COLOR, outline=PRIMARY_COLOR,
                                         width=2)
        self.text_id = self.create_text(width//2, height//2, text=text,
                                       fill=TEXT_COLOR, font=("Segoe UI", 12, "bold"))
        
        # Bind events
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        
    def on_enter(self, e):
        self.itemconfig(self.rect, fill=PRIMARY_COLOR)
        self.hovered = True
        
    def on_leave(self, e):
        self.itemconfig(self.rect, fill=ACCENT_COLOR)
        self.hovered = False
        
    def on_click(self, e):
        if self.command:
            self.command()


class SignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VSLearner - Sign Language Recognition")
        
        # Fullscreen setup
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg=BG_COLOR)
        
        # Get screen dimensions
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        
        self.running = False
        self.cap = None

        # model related
        self.tflite = None
        self.tf_model = None
        self.input_details = None
        self.output_details = None
        self.labels = []
        self.scaler = None
        self.use_tflite = False

        # mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.6,
                                         min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

        # ROI / state
        self.in_roi = False
        self.in_count = 0
        self.out_count = 0
        self.last_seen_time = 0.0

        # session state
        self.session_candidate = None
        self.session_candidate_count = 0
        self.session_last_appended = None
        self.lock = threading.Lock()

        # Statistics
        self.chars_count = 0
        self.words_count = 0
        self.confidence_history = []

        # result text
        self.tts_on = tk.BooleanVar(value=False)
        if TTS_AVAILABLE:
            self.tts_engine = pyttsx3.init()
        else:
            self.tts_engine = None

        # build UI
        self._build_ui()

        # try load model & labels & scaler
        self._load_labels()
        self._load_scaler()
        self._load_model()
        
        # Bind ESC key to exit fullscreen
        self.root.bind("<Escape>", self.toggle_fullscreen)
        self.root.bind("<F11>", self.toggle_fullscreen)

    def toggle_fullscreen(self, event=None):
        current = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not current)

    def _build_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Top header bar
        header_frame = tk.Frame(main_frame, bg=SECONDARY_BG, height=80)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(header_frame, text="VSLearner", 
                              font=("Segoe UI", 32, "bold"),
                              fg=PRIMARY_COLOR, bg=SECONDARY_BG)
        title_label.pack(side=tk.LEFT, padx=30, pady=15)
        
        subtitle_label = tk.Label(header_frame, text="Sign Language Recognition System",
                                 font=("Segoe UI", 14),
                                 fg=TEXT_COLOR, bg=SECONDARY_BG)
        subtitle_label.pack(side=tk.LEFT, padx=(0, 30), pady=15)
        
        # Status indicator
        self.status_indicator = tk.Canvas(header_frame, width=20, height=20,
                                         bg=SECONDARY_BG, highlightthickness=0)
        self.status_indicator.pack(side=tk.RIGHT, padx=30)
        self.status_circle = self.status_indicator.create_oval(2, 2, 18, 18,
                                                              fill=WARNING_COLOR,
                                                              outline="")
        
        # Main content area - 3 columns
        content_frame = tk.Frame(main_frame, bg=BG_COLOR)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video feed (60% width)
        left_frame = tk.Frame(content_frame, bg=SECONDARY_BG)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video header
        video_header = tk.Frame(left_frame, bg=ACCENT_COLOR, height=50)
        video_header.pack(fill=tk.X)
        video_header.pack_propagate(False)
        
        tk.Label(video_header, text="📹 CAMERA FEED", 
                font=("Segoe UI", 14, "bold"),
                fg=TEXT_COLOR, bg=ACCENT_COLOR).pack(pady=12)
        
        # Video display
        self.video_panel = tk.Label(left_frame, bg="#000000")
        self.video_panel.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Right panel - Controls and output
        right_frame = tk.Frame(content_frame, bg=BG_COLOR, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Control panel
        control_frame = tk.Frame(right_frame, bg=SECONDARY_BG)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(control_frame, text="⚙️ CONTROLS", 
                font=("Segoe UI", 14, "bold"),
                fg=TEXT_COLOR, bg=SECONDARY_BG).pack(pady=(15, 10))
        
        # Buttons
        btn_frame = tk.Frame(control_frame, bg=SECONDARY_BG)
        btn_frame.pack(pady=10)
        
        self.start_btn = ModernButton(btn_frame, "▶ START CAMERA", 
                                      self.toggle_camera, width=340, height=55)
        self.start_btn.pack(pady=8, padx=20)
        
        clear_btn = ModernButton(btn_frame, "🗑 CLEAR TEXT", 
                                self.clear_text, width=340, height=50)
        clear_btn.pack(pady=8, padx=20)
        
        save_btn = ModernButton(btn_frame, "💾 SAVE TO FILE", 
                               self.save_text, width=340, height=50)
        save_btn.pack(pady=8, padx=20)
        
        # TTS Toggle
        tts_frame = tk.Frame(control_frame, bg=SECONDARY_BG)
        tts_frame.pack(pady=15)
        
        tk.Checkbutton(tts_frame, text="🔊 Text-to-Speech", 
                      variable=self.tts_on,
                      font=("Segoe UI", 11),
                      fg=TEXT_COLOR, bg=SECONDARY_BG,
                      selectcolor=ACCENT_COLOR,
                      activebackground=SECONDARY_BG,
                      activeforeground=PRIMARY_COLOR).pack()
        
        # Statistics panel
        stats_frame = tk.Frame(right_frame, bg=SECONDARY_BG)
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(stats_frame, text="📊 STATISTICS", 
                font=("Segoe UI", 14, "bold"),
                fg=TEXT_COLOR, bg=SECONDARY_BG).pack(pady=(15, 10))
        
        stats_content = tk.Frame(stats_frame, bg=SECONDARY_BG)
        stats_content.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # Stats labels
        self.chars_label = tk.Label(stats_content, text="Characters: 0",
                                    font=("Segoe UI", 11),
                                    fg=TEXT_COLOR, bg=SECONDARY_BG,
                                    anchor='w')
        self.chars_label.pack(fill=tk.X, pady=3)
        
        self.words_label = tk.Label(stats_content, text="Words: 0",
                                    font=("Segoe UI", 11),
                                    fg=TEXT_COLOR, bg=SECONDARY_BG,
                                    anchor='w')
        self.words_label.pack(fill=tk.X, pady=3)
        
        self.conf_label = tk.Label(stats_content, text="Avg Confidence: --",
                                   font=("Segoe UI", 11),
                                   fg=TEXT_COLOR, bg=SECONDARY_BG,
                                   anchor='w')
        self.conf_label.pack(fill=tk.X, pady=3)
        
        # Output text panel
        output_frame = tk.Frame(right_frame, bg=SECONDARY_BG)
        output_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(output_frame, text="📝 RECOGNIZED TEXT", 
                font=("Segoe UI", 14, "bold"),
                fg=TEXT_COLOR, bg=SECONDARY_BG).pack(pady=(15, 10))
        
        # Text widget with custom styling
        text_container = tk.Frame(output_frame, bg=ACCENT_COLOR, padx=2, pady=2)
        text_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.result_widget = tk.Text(text_container, wrap='word',
                                     font=("Consolas", 12),
                                     bg="#0a0a0a", fg=SUCCESS_COLOR,
                                     insertbackground=PRIMARY_COLOR,
                                     relief=tk.FLAT,
                                     padx=15, pady=15)
        self.result_widget.pack(fill=tk.BOTH, expand=True)
        
        # Bottom status bar
        status_frame = tk.Frame(main_frame, bg=SECONDARY_BG, height=40)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(15, 0))
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="⚪ Ready - Press START CAMERA to begin")
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               font=("Segoe UI", 10),
                               fg=TEXT_COLOR, bg=SECONDARY_BG,
                               anchor='w')
        status_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Exit button
        exit_label = tk.Label(status_frame, text="Press ESC to exit fullscreen | F11 to toggle",
                            font=("Segoe UI", 9),
                            fg="#888888", bg=SECONDARY_BG)
        exit_label.pack(side=tk.RIGHT, padx=20, pady=10)

    def update_stats(self):
        """Update statistics display"""
        text = self.result_widget.get('1.0', tk.END).strip()
        self.chars_count = len(text.replace(' ', ''))
        self.words_count = len(text.split())
        
        self.chars_label.config(text=f"Characters: {self.chars_count}")
        self.words_label.config(text=f"Words: {self.words_count}")
        
        if self.confidence_history:
            avg_conf = sum(self.confidence_history[-20:]) / len(self.confidence_history[-20:])
            self.conf_label.config(text=f"Avg Confidence: {avg_conf:.1%}")

    def toggle_camera(self):
        if not self.running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open camera device")
            return
        
        # Set camera resolution for better performance on RPi
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.running = True
        self.status_var.set("🟢 Camera active - Show hand gesture in ROI")
        self.status_indicator.itemconfig(self.status_circle, fill=SUCCESS_COLOR)
        self.start_btn.itemconfig(self.start_btn.text_id, text="⏸ STOP CAMERA")
        self.last_seen_time = 0.0
        self._video_loop()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_var.set("🔴 Camera stopped")
        self.status_indicator.itemconfig(self.status_circle, fill=WARNING_COLOR)
        self.start_btn.itemconfig(self.start_btn.text_id, text="▶ START CAMERA")

    def clear_text(self):
        self.result_widget.delete('1.0', tk.END)
        self.chars_count = 0
        self.words_count = 0
        self.confidence_history = []
        self.update_stats()
        self.status_var.set("🗑 Text cleared")

    def save_text(self):
        txt = self.result_widget.get('1.0', tk.END).strip()
        if not txt:
            messagebox.showinfo("Save", "No content to save")
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".txt", 
                                            filetypes=[("Text files","*.txt")])
        if fpath:
            with open(fpath, 'w', encoding='utf8') as f:
                f.write(txt)
            messagebox.showinfo("Save", f"Saved to {fpath}")
            self.status_var.set(f"💾 Saved to {os.path.basename(fpath)}")

    def _load_labels(self):
        try:
            with open(LABELS_JSON, 'r', encoding='utf8') as f:
                j = json.load(f)
                self.labels = j.get('classes', [])
            self.status_var.set(f"✓ Labels loaded: {len(self.labels)} classes")
        except Exception as e:
            self.labels = []
            self.status_var.set(f"⚠ Labels not found: {e}")

    def _load_scaler(self):
        if SCALER_AVAILABLE and os.path.exists(SCALER_PKL):
            try:
                with open(SCALER_PKL, 'rb') as f:
                    self.scaler = pickle.load(f)
            except Exception as e:
                print("Scaler load error:", e)
                self.scaler = None

    def _load_model(self):
        if TFLITE_AVAILABLE and os.path.exists(MODEL_TFLITE):
            try:
                self.tflite = Interpreter(model_path=MODEL_TFLITE)
                self.tflite.allocate_tensors()
                self.input_details = self.tflite.get_input_details()
                self.output_details = self.tflite.get_output_details()
                self.use_tflite = True
                self.status_var.set("✓ TFLite model loaded successfully")
                return
            except Exception as e:
                print("TFLite load fail:", e)

        if 'tf' in globals() and os.path.exists(MODEL_H5):
            try:
                import tensorflow as tf
                self.tf_model = tf.keras.models.load_model(MODEL_H5)
                self.status_var.set("✓ Keras model loaded successfully")
                return
            except Exception as e:
                print("Keras load fail:", e)

        self.status_var.set("⚠ No model loaded (demo mode)")

    def _predict_keypoints(self, kp):
        x = np.array(kp, dtype=np.float32).reshape(1, -1)
        if self.scaler is not None:
            try:
                x = self.scaler.transform(x)
            except Exception as e:
                print("Scaler transform error", e)

        if self.use_tflite and self.tflite is not None:
            try:
                idx = self.input_details[0]['index']
                dtype = self.input_details[0]['dtype']
                inp = x.astype(dtype)
                self.tflite.set_tensor(idx, inp)
                self.tflite.invoke()
                out = self.tflite.get_tensor(self.output_details[0]['index'])
                probs = out.flatten()
            except Exception as e:
                print("TFLite invoke error", e)
                probs = np.zeros(len(self.labels) or 1)
        elif self.tf_model is not None:
            probs = self.tf_model.predict(x, verbose=0).flatten()
        else:
            probs = np.random.rand(len(self.labels) or 1)
            probs = probs / probs.sum()
        
        idx = int(np.argmax(probs))
        conf = float(probs[idx]) if probs.size > 0 else 0.0
        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        return label, conf

    def _video_loop(self):
        if not self.running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self._video_loop)
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ROI calculation
        side = int(min(h, w) * ROI_RATIO)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - side//2, cy - side//2
        x2, y2 = cx + side//2, cy + side//2

        # Process with mediapipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        hand_in_this_frame = False
        predicted_label = None
        predicted_conf = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks with custom style
            for connection in self.mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                start = hand_landmarks.landmark[start_idx]
                end = hand_landmarks.landmark[end_idx]
                
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
            
            # Draw landmarks
            for lm in hand_landmarks.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 4, (255, 0, 255), -1)
                cv2.circle(frame, (px, py), 5, (0, 255, 255), 1)

            # Wrist check
            wrist = hand_landmarks.landmark[0]
            wx = int(wrist.x * w)
            wy = int(wrist.y * h)
            cv2.circle(frame, (wx, wy), 8, (0, 255, 0), -1)

            if x1 <= wx <= x2 and y1 <= wy <= y2:
                hand_in_this_frame = True
                self.last_seen_time = time.time()
                
                kp = []
                for lm in hand_landmarks.landmark:
                    kp.extend([lm.x, lm.y, lm.z])
                
                label, conf = self._predict_keypoints(kp)
                predicted_label = label
                predicted_conf = conf
                self.confidence_history.append(conf)
                
                # Display prediction with modern style
                text = f"{label}"
                conf_text = f"{conf:.1%}"
                
                # Background for text
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                cv2.rectangle(frame, (x1, y1-60), (x1+tw+120, y1-5), (0, 0, 0), -1)
                cv2.rectangle(frame, (x1, y1-60), (x1+tw+120, y1-5), (0, 212, 255), 2)
                
                cv2.putText(frame, text, (x1+10, y1-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
                cv2.putText(frame, conf_text, (x1+10, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 136), 2)

        # Draw ROI with modern styling
        # Corner style ROI
        corner_length = 30
        thickness = 3
        color = (0, 255, 255) if hand_in_this_frame else (0, 200, 200)
        
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
        
        # ROI label
        cv2.putText(frame, "DETECTION ZONE", (x1, y2+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update state logic (same as original)
        if hand_in_this_frame:
            self.in_count += 1
            self.out_count = 0
        else:
            self.out_count += 1
            self.in_count = 0

        just_entered = False
        just_exited = False
        now = time.time()

        if self.in_count >= MIN_IN_FRAMES and not self.in_roi:
            self.in_roi = True
            just_entered = True
            with self.lock:
                self.session_candidate = None
                self.session_candidate_count = 0
                self.session_last_appended = None

        time_since_last_seen_ms = (now - self.last_seen_time) * 1000.0 if self.last_seen_time else 9999.0
        if (self.out_count >= MIN_IN_FRAMES or time_since_last_seen_ms > EXIT_TIMEOUT_MS) and self.in_roi:
            self.in_roi = False
            just_exited = True

        if self.in_roi:
            if predicted_label is not None:
                if self.session_candidate != predicted_label:
                    self.session_candidate = predicted_label
                    self.session_candidate_count = 1
                else:
                    self.session_candidate_count += 1
                
                if self.session_candidate_count >= STABLE_FRAMES:
                    if self.session_last_appended != self.session_candidate:
                        current = self.result_widget.get('1.0', tk.END).rstrip('\n')
                        newtxt = (current + self.session_candidate) if current else self.session_candidate
                        self.result_widget.delete('1.0', tk.END)
                        self.result_widget.insert('1.0', newtxt)
                        self.update_stats()
                        self.status_var.set(f"✓ Recognized: {self.session_candidate} ({predicted_conf:.1%})")
                        
                        if self.tts_on.get() and self.tts_engine:
                            try:
                                self.tts_engine.say(self.session_candidate)
                                self.tts_engine.runAndWait()
                            except Exception:
                                pass
                        
                        self.session_last_appended = self.session_candidate
                        self.session_candidate_count = 0
                        self.session_candidate = None

        if just_exited:
            current = self.result_widget.get('1.0', tk.END).rstrip('\n')
            should_space = False
            
            if self.session_last_appended is not None:
                should_space = True
            elif current and not current.endswith(' '):
                should_space = True

            if should_space and not current.endswith(' '):
                newtxt = current + " "
                self.result_widget.delete('1.0', tk.END)
                self.result_widget.insert('1.0', newtxt)
                self.update_stats()
                self.status_var.set("📝 Word completed")
            
            with self.lock:
                self.session_candidate = None
                self.session_candidate_count = 0
                self.session_last_appended = None

        # Convert and display frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        
        # Resize to fit panel while maintaining aspect ratio
        panel_width = self.video_panel.winfo_width()
        panel_height = self.video_panel.winfo_height()
        
        if panel_width > 1 and panel_height > 1:
            im_pil.thumbnail((panel_width, panel_height), Image.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=im_pil)
        self.video_panel.imgtk = imgtk
        self.video_panel.config(image=imgtk)

        self.root.after(10, self._video_loop)



def main():
    root = tk.Tk()
    app = SignApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
