import cv2
import mediapipe as mp
import json
import os
import time


SAVE_DIR = "../data/raw/movement_data"

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # Công cụ để vẽ
mp_drawing_styles = mp.solutions.drawing_styles 

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def format_landmarks(hand_landmarks):
    coords = []
    for p in hand_landmarks.landmark:
        coords.extend([p.x, p.y, p.z])
    return coords # vector 63 values

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    cap = cv2.VideoCapture(0)
    recording = False
    sequence = []
    
    # Nhập label trước khi mở camera
    label = input("Nhập tên ký hiệu movement: ").strip()
    label_dir = os.path.join(SAVE_DIR, label)
    ensure_dir(label_dir)

    print(f"Label: {label}")
    print("Nhấn 'S' để bắt đầu ghi. Nhấn 'E' để kết thúc và lưu. ESC để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1. Xử lý MediaPipe (Chuyển BGR sang RGB)
        frame.flags.writeable = False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        frame.flags.writeable = True

        # Hiển thị thông tin Label
        cv2.putText(frame, f"Label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 2. Nếu phát hiện tay
        current_landmarks_list = None 
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # --- VẼ LANDMARKS LÊN MÀN HÌNH ---
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Chuẩn bị dữ liệu để lưu (nếu đang ghi)
                current_landmarks_list = format_landmarks(hand_landmarks)

        # 3. Logic Ghi hình
        if recording:
            cv2.putText(frame, "Recording...", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Chỉ lưu nếu phát hiện được tay
            if current_landmarks_list is not None:
                sequence.append(current_landmarks_list)

        cv2.imshow("Movement Recorder", frame)
        key = cv2.waitKey(1)

        # Xử lý phím bấm
        if key == ord('s'):
            print("Bắt đầu ghi...")
            recording = True
            sequence = []

        if key == ord('e'):
            if recording and len(sequence) > 5:
                filename = f"{int(time.time())}.json"
                filepath = os.path.join(label_dir, filename)

                data = {
                    "label": label,
                    "sequence": sequence
                }

                with open(filepath, "w") as f:
                    json.dump(data, f)

                print(f"Đã lưu: {filepath} (frames = {len(sequence)})")
            elif recording:
                print("Dữ liệu quá ngắn hoặc không tìm thấy tay, không lưu.")
            else:
                print("Chưa ở chế độ ghi.")

            recording = False
            sequence = []

        if key == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()