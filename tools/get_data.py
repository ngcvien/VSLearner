import cv2
import mediapipe as mp
import csv
import os
import time

# --- Khởi tạo MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# --- Nhập nhãn ký hiệu ---
label = input("Nhập tên ký hiệu (ví dụ: Hello, A, B...): ")

# --- Tạo folder riêng cho ký hiệu ---
folder_path = f"../data/raw/{label}"
os.makedirs(folder_path, exist_ok=True)

# --- Tìm số file hiện có để đặt tên tiếp theo ---
existing = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
file_index = len(existing) + 1
csv_path = os.path.join(folder_path, f"{label}_{file_index}.csv")

# --- Mở webcam ---
cap = cv2.VideoCapture(0)
print(f"📸 Bắt đầu thu dữ liệu cho ký hiệu '{label}' (file {csv_path})")
print("Nhấn 'q' để dừng.")
time.sleep(2)

count = 0
with open(csv_path, mode='w', newline='') as f:
    csv_writer = csv.writer(f)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                data_row = []
                for lm in hand_landmarks.landmark:
                    data_row += [lm.x, lm.y, lm.z]
                data_row.append(label)
                csv_writer.writerow(data_row)
                count += 1

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Samples: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Thu thập ký hiệu", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"✅ Đã lưu {count} mẫu vào {csv_path}")
