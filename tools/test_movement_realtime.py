import cv2
import numpy as np
import mediapipe as mp
try:
    # Raspberry Pi / Linux
    import tflite_runtime.interpreter as tflite
    print("Using tflite_runtime (Raspberry Pi / Linux).")
except ImportError:
    # Windows / Mac / PC
    import tensorflow as tf
    tflite = tf.lite
    print("Using TensorFlow Lite interpreter (Windows/PC).")
import json
import time

# ================================================================
# CONFIG
# ================================================================
MODEL_PATH = "../models/movement_model.tflite"
LABEL_PATH = "../models/movement_labels.json"

SEQ_LEN = 100
START_TH = 0.008
STOP_TH  = 0.004

last_text = ""
last_time = 0
DISPLAY_TIME = 1.5


# ================================================================
# MOVEMENT DETECTOR (NO GLOBALS)
# ================================================================
class MovementDetector:
    def __init__(self, model_path, seq_len):
        self.seq_len = seq_len

        self.prev = None
        self.buffer = []
        self.state = "IDLE"

        # --- Anti jitter ---
        self.diff_hist = []
        self.DIFF_WIN = 5

        self.start_counter = 0
        self.stop_counter = 0

        self.START_FRAMES = 6      # cần ≥ 6 frame liên tục
        self.STOP_FRAMES = 10      # cần ≥ 10 frame liên tục
        self.MIN_RECORD_FRAMES = 30

        self.cooldown_until = 0
        self.COOLDOWN_TIME = 0.6   # giây

        # Load TFLite
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_idx = self.interpreter.get_input_details()[0]["index"]
        self.output_idx = self.interpreter.get_output_details()[0]["index"]

    def calc_diff(self, cur):
        if self.prev is None:
            self.prev = cur
            return 0.0

        diff = np.mean(np.abs(cur - self.prev))
        self.prev = cur

        # moving average
        self.diff_hist.append(diff)
        if len(self.diff_hist) > self.DIFF_WIN:
            self.diff_hist.pop(0)

        return np.mean(self.diff_hist)

    def normalize_sequence(self, seq):
        idx = np.linspace(0, len(seq) - 1, self.seq_len)
        return np.array([seq[int(i)] for i in idx], dtype=np.float32)

    def predict(self, seq):
        seq = self.normalize_sequence(seq)
        seq = np.expand_dims(seq, axis=0)

        self.interpreter.set_tensor(self.input_idx, seq.astype(np.float32))
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_idx)[0]

        return int(np.argmax(out)), float(np.max(out))

    def update(self, lm_vec):
        now = time.time()
        if now < self.cooldown_until:
            return None

        lm_vec = np.array(lm_vec, dtype=np.float32)
        diff = self.calc_diff(lm_vec)

        # =========================
        # IDLE → RECORDING
        # =========================
        if self.state == "IDLE":
            if diff > START_TH:
                self.start_counter += 1
                if self.start_counter >= self.START_FRAMES:
                    self.state = "RECORDING"
                    self.buffer = []
                    self.start_counter = 0
                    print("[START Movement]")
            else:
                self.start_counter = 0

        # =========================
        # RECORDING
        # =========================
        if self.state == "RECORDING":
            self.buffer.append(lm_vec)

            if diff < STOP_TH:
                self.stop_counter += 1
            else:
                self.stop_counter = 0

            if (
                self.stop_counter >= self.STOP_FRAMES and
                len(self.buffer) >= self.MIN_RECORD_FRAMES
            ):
                print("[STOP Movement]")
                self.state = "IDLE"
                self.stop_counter = 0
                self.cooldown_until = time.time() + self.COOLDOWN_TIME

                pred = self.predict(self.buffer)
                self.buffer = []
                return pred

        return None


# ================================================================
# LOAD LABELS
# ================================================================
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)


# ================================================================
# MEDIA PIPE
# ================================================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ================================================================
# RUN TEST
# ================================================================
detector = MovementDetector(MODEL_PATH, SEQ_LEN)

cap = cv2.VideoCapture(0)
print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    h, w, c = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        lm_vec = []
        for p in hand.landmark:
            lm_vec.extend([p.x, p.y, p.z])

        res = detector.update(lm_vec)

        if res is not None:
            pred_idx, conf = res
            text = f"{LABELS[pred_idx]} ({conf:.2f})"
            cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    cv2.putText(img, f"State: {detector.state}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Movement Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
