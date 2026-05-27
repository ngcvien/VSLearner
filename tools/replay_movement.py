import json
import cv2
import numpy as np
import os
import glob

# --- CẤU HÌNH ---
DATA_DIR = "../data/raw/movement_data"
WIDTH, HEIGHT = 640, 480  # Kích thước màn hình mô phỏng

# Định nghĩa các cặp điểm nối với nhau để tạo thành hình bàn tay
# (MediaPipe Hand Connections)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),       # Ngón trỏ
    (9, 10), (10, 11), (11, 12),          # Ngón giữa (nối 0-9 ở dưới)
    (13, 14), (14, 15), (15, 16),         # Ngón áp út (nối 0-13 ở dưới)
    (0, 17), (17, 18), (18, 19), (19, 20),# Ngón út
    (5, 9), (9, 13), (13, 17)             # Lòng bàn tay
]

def draw_hand_from_landmarks(canvas, landmarks):
    """
    Vẽ bàn tay lên canvas từ danh sách 63 giá trị (21 điểm * 3 coords)
    """
    h, w, _ = canvas.shape
    points = []

    # 1. Chuyển đổi list phẳng thành các điểm (x, y) pixel
    # landmarks format: [x0, y0, z0, x1, y1, z1, ...]
    for i in range(0, len(landmarks), 3):
        x = landmarks[i]
        y = landmarks[i+1]
        # z = landmarks[i+2] # Chúng ta bỏ qua Z khi vẽ 2D
        
        px, py = int(x * w), int(y * h)
        points.append((px, py))
        
        # Vẽ các khớp (điểm tròn)
        cv2.circle(canvas, (px, py), 4, (0, 255, 255), -1) # Màu vàng

    # 2. Vẽ các đường nối (xương tay)
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        if p1_idx < len(points) and p2_idx < len(points):
            pt1 = points[p1_idx]
            pt2 = points[p2_idx]
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 2) # Màu xanh lá

def select_file():
    """Hàm hỗ trợ liệt kê file để chọn"""
    # Tìm tất cả file json trong thư mục con
    files = glob.glob(os.path.join(DATA_DIR, "*", "*.json"))
    
    if not files:
        print("Không tìm thấy file dữ liệu nào!")
        return None

    print("\n--- DANH SÁCH CÁC FILE DỮ LIỆU ---")
    for i, f in enumerate(files):
        print(f"{i}: {f}")
    
    try:
        idx = int(input("\nChọn số thứ tự file muốn xem: "))
        return files[idx]
    except (ValueError, IndexError):
        print("Lựa chọn không hợp lệ.")
        return None

def main():
    filepath = select_file()
    if not filepath:
        return

    # Đọc dữ liệu
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    label = data['label']
    sequence = data['sequence']
    
    print(f"\nĐang phát lại: {label}")
    print(f"Tổng số frame: {len(sequence)}")
    print("Nhấn 'Q' hoặc ESC để thoát sớm.")

    # Vòng lặp phát lại
    for i, frame_landmarks in enumerate(sequence):
        # 1. Tạo màn hình đen
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # 2. Vẽ tay
        draw_hand_from_landmarks(canvas, frame_landmarks)
        
        # 3. Hiển thị thông tin
        cv2.putText(canvas, f"Label: {label}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, f"Frame: {i}/{len(sequence)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Data Replay", canvas)
        
        # Điều chỉnh tốc độ (30ms ~ 33fps, giống tốc độ webcam)
        key = cv2.waitKey(40) 
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    print("Hoàn tất phát lại.")

if __name__ == "__main__":
    main()