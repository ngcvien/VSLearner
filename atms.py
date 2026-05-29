#!/usr/bin/env python3
"""
atms.py

Adaptive Temporal Movement Segmentation (ATMS)

Công thức:
Mt = sqrt(sum((Lt(i) - Lt-1(i))^2))

START movement:
Mt > Tm trong Ns frame liên tục

STOP movement:
Mt <= Te trong Ne frame liên tục
"""

import os
import json
import time
import platform
from dataclasses import dataclass
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    try:
        import tensorflow as tf
        tflite = tf.lite
    except Exception:
        tflite = None


def is_raspberry_pi():
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system != "linux" or not machine.startswith(("arm", "aarch64")):
        return False

    model_paths = (
        "/proc/device-tree/model",
        "/sys/firmware/devicetree/base/model",
    )

    for model_path in model_paths:
        try:
            with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
                if "raspberry pi" in f.read().lower():
                    return True
        except OSError:
            pass

    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
            cpuinfo = f.read().lower()
    except OSError:
        return False

    return "raspberry pi" in cpuinfo or "raspberrypi" in cpuinfo


def detect_atms_profile():
    return "pi" if is_raspberry_pi() else "laptop"


def load_atms_config(profile=None, path="configs/atms_profiles.json"):
    profile = profile or os.getenv("ATMS_PROFILE") or detect_atms_profile()

    with open(path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    if profile not in profiles:
        raise ValueError(f"Không có profile ATMS: {profile}")

    return ATMSConfig(**profiles[profile])

@dataclass
class ATMSConfig:
    seq_len: int = 100

    start_threshold: float = 0.050
    stop_threshold: float = 0.025

    start_frames: int = 4
    stop_frames: int = 8

    min_record_frames: int = 20
    max_record_frames: int = 180

    smooth_window: int = 5
    pre_roll_frames: int = 6

    cooldown_sec: float = 0.6


class ATMS:
    def __init__(self, config=None):
        self.cfg = config or ATMSConfig()
        self.reset()

    def reset(self):
        self.state = "IDLE"
        self.prev = None

        self.buffer = []
        self.pre_roll = []

        self.score_hist = []

        self.start_counter = 0
        self.stop_counter = 0

        self.cooldown_until = 0.0
        self.last_score = 0.0

    def _to_vector(self, lm_vec):
        vec = np.asarray(lm_vec, dtype=np.float32).reshape(-1)

        if vec.shape[0] != 63:
            raise ValueError(f"ATMS cần vector 63 chiều, hiện tại: {vec.shape[0]}")

        return vec

    def movement_score(self, cur):
        """
        Mt = sqrt(sum((Lt - Lt-1)^2))
        """

        if self.prev is None:
            self.prev = cur.copy()
            return 0.0

        score = float(np.linalg.norm(cur - self.prev))
        self.prev = cur.copy()

        self.score_hist.append(score)

        if len(self.score_hist) > self.cfg.smooth_window:
            self.score_hist.pop(0)

        return float(np.mean(self.score_hist))

    def update(self, lm_vec):
        now = time.time()

        if now < self.cooldown_until:
            return None

        cur = self._to_vector(lm_vec)
        mt = self.movement_score(cur)

        self.last_score = mt

        if self.state == "IDLE":
            self.pre_roll.append(cur.copy())

            if len(self.pre_roll) > self.cfg.pre_roll_frames:
                self.pre_roll.pop(0)

            if mt > self.cfg.start_threshold:
                self.start_counter += 1

                if self.start_counter >= self.cfg.start_frames:
                    self.state = "RECORDING"
                    self.buffer = [x.copy() for x in self.pre_roll]

                    self.start_counter = 0
                    self.stop_counter = 0

                    return {
                        "event": "start",
                        "score": mt
                    }
            else:
                self.start_counter = 0

            return None

        if self.state == "RECORDING":
            self.buffer.append(cur.copy())

            if mt <= self.cfg.stop_threshold:
                self.stop_counter += 1
            else:
                self.stop_counter = 0

            enough_stop = self.stop_counter >= self.cfg.stop_frames
            enough_len = len(self.buffer) >= self.cfg.min_record_frames
            forced_stop = len(self.buffer) >= self.cfg.max_record_frames

            if (enough_stop and enough_len) or forced_stop:
                seq = np.asarray(self.buffer, dtype=np.float32)
                frames = len(seq)

                self.state = "IDLE"
                self.buffer = []
                self.pre_roll = []

                self.start_counter = 0
                self.stop_counter = 0

                self.cooldown_until = time.time() + self.cfg.cooldown_sec

                if frames < self.cfg.min_record_frames:
                    return {
                        "event": "discard",
                        "frames": frames,
                        "score": mt
                    }

                return {
                    "event": "stop",
                    "sequence": seq,
                    "frames": frames,
                    "score": mt,
                    "forced": forced_stop
                }

        return None


def normalize_sequence(seq, target_len=100):
    seq = np.asarray(seq, dtype=np.float32)

    if seq.ndim != 2 or seq.shape[1] != 63:
        raise ValueError(f"Sequence phải có shape (frames, 63), hiện tại: {seq.shape}")

    n = seq.shape[0]

    if n == target_len:
        return seq.astype(np.float32)

    old_idx = np.linspace(0, 1, n)
    new_idx = np.linspace(0, 1, target_len)

    out = np.zeros((target_len, 63), dtype=np.float32)

    for j in range(63):
        out[:, j] = np.interp(new_idx, old_idx, seq[:, j])

    return out


class MovementRecognizer:
    def __init__(
        self,
        model_path="models/movement_model.tflite",
        label_path="models/movement_labels.json",
        config=None
    ):
        self.atms = ATMS(config)
        self.seq_len = self.atms.cfg.seq_len

        self.labels = self._load_labels(label_path)

        if tflite is None:
            raise RuntimeError("Không tìm thấy tflite_runtime hoặc tensorflow.")

        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_idx = self.interpreter.get_input_details()[0]["index"]
        self.output_idx = self.interpreter.get_output_details()[0]["index"]

    def _load_labels(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data.get("classes", data.get("labels", []))

        return data

    @property
    def state(self):
        return self.atms.state

    @property
    def last_score(self):
        return self.atms.last_score

    def predict_sequence(self, seq):
        x = normalize_sequence(seq, self.seq_len)
        x = np.expand_dims(x, axis=0).astype(np.float32)

        self.interpreter.set_tensor(self.input_idx, x)
        self.interpreter.invoke()

        out = self.interpreter.get_tensor(self.output_idx)[0]

        idx = int(np.argmax(out))
        conf = float(np.max(out))

        label = self.labels[idx] if idx < len(self.labels) else str(idx)

        return label, conf, idx

    def update(self, lm_vec):
        event = self.atms.update(lm_vec)

        if event is None:
            return None

        if event["event"] == "stop":
            label, conf, idx = self.predict_sequence(event["sequence"])

            event["label"] = label
            event["confidence"] = conf
            event["pred_idx"] = idx

        return event
