#!/usr/bin/env python3
# gui_app_pi.py — Raspberry Pi 4 + Mediapipe + TFLite + Tkinter

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import time
import json
import threading

try:
    import tflite_runtime.interpreter as tflite
except:
    import tensorflow as tf
    tflite = tf.lite

# =================== CONFIG ===================
ROI_X, ROI_Y = 100, 100
ROI_W, ROI_H = 300, 300

STABLE_FRAMES = 7
EXIT_TIMEOUT_MS = 600

CAM_WIDTH, CAM_HEIGHT = 640, 480
# ===============================================


class PiSignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Recognition — Raspberry Pi 4")

        # Camera View
        self.canvas = tk.Label(root)
        self.canvas.pack()

        # Output box
        self.output = tk.Text(root, height=5, width=60, font=("Arial", 16))
        self.output.pack(pady=8)

        # Load TFLite
        self.interpreter = tflite.Interpreter("models/sign_model.tflite")
        self.interpreter.allocate_tensors()
        self.in_details = self.interpreter.get_input_details()
        self.out_details = self.interpreter.get_output_details()

        # Load labels
        with open("models/labels.json", "r") as f:
            self.labels = json.load(f)

        # Mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Camera
        url = "http://192.168.1.177:4747/video"
        self.cap = cv2.VideoCapture(url)
        self.cap.set(3, CAM_WIDTH)
        self.cap.set(4, CAM_HEIGHT)

        # State
        self.last_seen = 0
        self.prev_char = None
        self.stable_count = 0
        self.pending_space = False

        self.update_frame()

    # ------------------------------------------------
    def extract_landmarks(self, frame):
        """Run mediapipe and return 63 floats or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0]
        coords = []
        for p in lm.landmark:
            coords.extend([p.x, p.y, p.z])  # 21 landmarks × 3 = 63
        return np.array(coords, dtype=np.float32)

    # ------------------------------------------------
    def run_model(self, keypoints):
        """TFLite inference for 63 landmark inputs."""
        x = keypoints.reshape(1, 63).astype("float32")
        self.interpreter.set_tensor(self.in_details[0]['index'], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.out_details[0]['index'])[0]
        return np.argmax(out)

    # ------------------------------------------------
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        # ROI rectangle
        cv2.rectangle(frame, (ROI_X, ROI_Y),
                      (ROI_X + ROI_W, ROI_Y + ROI_H),
                      (0, 255, 0), 2)

        # Crop ROI
        roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        now = time.time() * 1000  # ms

        # Extract hand landmarks
        keypoints = self.extract_landmarks(roi)

        if keypoints is not None:
            # Hand detected inside ROI
            self.last_seen = now
            self.pending_space = False

            try:
                idx = self.run_model(keypoints)
                ch = self.labels[idx]
            except:
                ch = ""

            # Stability filter
            if ch == self.prev_char:
                self.stable_count += 1
            else:
                self.stable_count = 0

            if self.stable_count >= STABLE_FRAMES:
                self.add_char(ch)
                self.stable_count = 0

            self.prev_char = ch

        else:
            # Hand missing
            if now - self.last_seen > EXIT_TIMEOUT_MS and not self.pending_space:
                self.output.insert(tk.END, " ")
                self.output.see(tk.END)
                self.pending_space = True
                self.prev_char = None

        # Render
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tk_img = ImageTk.PhotoImage(img)

        self.canvas.configure(image=tk_img)
        self.canvas.image = tk_img

        self.root.after(28, self.update_frame)  # ~30 FPS

    # ------------------------------------------------
    def add_char(self, ch):
        if not ch.strip():
            return
        self.output.insert(tk.END, ch)
        self.output.see(tk.END)


# ----------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PiSignApp(root)
    root.mainloop()
