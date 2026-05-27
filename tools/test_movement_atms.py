#!/usr/bin/env python3
"""
tools/test_movement_atms.py

Test realtime:
MediaPipe → ATMS → movement_model.tflite → label

Logic chống nhận nhầm reset tay:
- Movement 1  → bỏ qua
- Movement 2  → hiển thị
- Movement 3  → bỏ qua
- Movement 4  → hiển thị
"""

import os
import sys
import cv2
import mediapipe as mp

sys.path.append(os.path.abspath(".."))

from atms import ATMSConfig, MovementRecognizer


MODEL_PATH = "../models/movement_model.tflite"
LABEL_PATH = "../models/movement_labels.json"


CONFIG = ATMSConfig(
    seq_len=100,
    start_threshold=0.050,
    stop_threshold=0.025,
    start_frames=4,
    stop_frames=8,
    min_record_frames=20,
    max_record_frames=180,
    smooth_window=5,
    pre_roll_frames=6,
    cooldown_sec=0.6
)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def landmarks_to_vector(hand_landmarks):
    vec = []

    for p in hand_landmarks.landmark:
        vec.extend([p.x, p.y, p.z])

    return vec


def main():
    recognizer = MovementRecognizer(
        model_path=MODEL_PATH,
        label_path=LABEL_PATH,
        config=CONFIG
    )

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Không mở được camera")
        return

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    last_text = ""

    # True nghĩa là movement kế tiếp sẽ bị bỏ qua.
    # Theo yêu cầu: bắt đầu bỏ từ hành động 1.
    skip_next_movement = True

    print("ATMS realtime test.")
    print("Logic: Movement 1 SKIP, Movement 2 SHOW, Movement 3 SKIP, Movement 4 SHOW...")
    print("Nhấn Q để thoát.")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("[WARN] Không đọc được frame")
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            lm_vec = landmarks_to_vector(hand)

            event = recognizer.update(lm_vec)

            if event is not None:
                if event["event"] == "start":
                    print("[START Movement]")

                elif event["event"] == "stop":
                    label = event["label"]
                    conf = event["confidence"]

                    if skip_next_movement:
                        print(f"[SKIP] {label} ({conf:.2f})")
                        skip_next_movement = False
                    else:
                        last_text = f"{label} ({conf:.2f})"
                        print("[SHOW]", last_text)
                        skip_next_movement = True

        cv2.putText(
            frame,
            f"State: {recognizer.state}",
            (10, h - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Mt: {recognizer.last_score:.4f}",
            (10, h - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        mode_text = "Next: SKIP" if skip_next_movement else "Next: SHOW"

        cv2.putText(
            frame,
            mode_text,
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        if last_text:
            cv2.putText(
                frame,
                last_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

        cv2.imshow("ATMS Movement Realtime Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Nhấn R để reset lại trạng thái bỏ qua
        if key == ord("r"):
            skip_next_movement = True
            last_text = ""
            recognizer.atms.reset()
            print("[RESET] Next movement will be skipped")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()