#!/usr/bin/env python3
"""
tools/record_movement_atms.py

Thu dữ liệu movement bằng ATMS có cơ chế ARM/DISARM.

SPACE: sẵn sàng thu mẫu mới
Q    : thoát chương trình

Quy trình:
1. Đưa tay về tư thế bắt đầu
2. Nhấn SPACE để ARM
3. Thực hiện movement
4. Dừng tay
5. Hệ thống tự SAVE và DISARM
6. Đưa tay về tư thế bắt đầu tiếp theo
7. Nhấn SPACE để thu mẫu tiếp theo
"""

import os
import sys
import json
import time
import cv2
import mediapipe as mp

sys.path.append(os.path.abspath(".."))

from atms import ATMS, load_atms_config


SAVE_DIR = "../data/raw/movement_data"
ATMS_CONFIG_PATH = "../configs/atms_profiles.json"


CONFIG = load_atms_config(path=ATMS_CONFIG_PATH)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def landmarks_to_vector(hand_landmarks):
    vec = []

    for p in hand_landmarks.landmark:
        vec.extend([p.x, p.y, p.z])

    return vec


def main():
    label = input("Nhập tên ký hiệu movement: ").strip()

    label_dir = os.path.join(SAVE_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    atms = ATMS(CONFIG)

    cap = cv2.VideoCapture(0)

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    saved_count = 0

    # Trạng thái điều khiển thu dữ liệu
    armed = False

    print("ATMS Recorder đang chạy.")
    print("SPACE: sẵn sàng thu mẫu mới")
    print("Q: thoát")
    print()
    print("Quy trình chuẩn:")
    print("1. Đưa tay về vị trí bắt đầu")
    print("2. Nhấn SPACE")
    print("3. Thực hiện movement")
    print("4. Dừng tay để hệ thống tự SAVE")
    print("5. Đưa tay về vị trí đầu rồi nhấn SPACE cho mẫu tiếp theo")

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

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

            # Chỉ cho ATMS xử lý khi đã ARM
            if armed:
                event = atms.update(lm_vec)

                if event is not None:
                    if event["event"] == "start":
                        print(f"[START] Mt = {event['score']:.4f}")

                    elif event["event"] == "stop":
                        filename = f"{int(time.time() * 1000)}.json"
                        filepath = os.path.join(label_dir, filename)

                        data = {
                            "label": label,
                            "segmentation": "ATMS",
                            "frames": int(event["frames"]),
                            "sequence": event["sequence"].tolist()
                        }

                        with open(filepath, "w", encoding="utf-8") as f:
                            json.dump(data, f)

                        saved_count += 1

                        print(f"[SAVE] {filepath} | frames = {event['frames']}")

                        # Quan trọng:
                        # Sau khi SAVE thì khóa lại để tránh ghi nhầm movement reset tay
                        armed = False
                        atms.reset()

                    elif event["event"] == "discard":
                        print(f"[DISCARD] frames = {event['frames']}")

                        armed = False
                        atms.reset()

        # Hiển thị trạng thái
        if armed:
            mode_text = "ARMED - thực hiện movement"
            mode_color = (0, 255, 0)
        else:
            mode_text = "DISARMED - dua tay ve vi tri dau, nhan SPACE"
            mode_color = (0, 0, 255)

        cv2.putText(
            frame,
            f"Label: {label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            mode_text,
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            mode_color,
            2
        )

        cv2.putText(
            frame,
            f"ATMS State: {atms.state} | Mt: {atms.last_score:.4f} | Saved: {saved_count}",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow("ATMS Movement Recorder", frame)

        key = cv2.waitKey(1) & 0xFF

        # SPACE để sẵn sàng thu mẫu mới
        if key == 32:
            armed = True
            atms.reset()
            print("[ARMED] Sẵn sàng thu movement mới")

        # Q để thoát
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
