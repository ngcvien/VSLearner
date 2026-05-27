import os
import json
import numpy as np

RAW_DIR = "../data/raw/movement_data"
OUTPUT_X = "../data/movement_X.npy"
OUTPUT_Y = "../data/movement_y.npy"
LABEL_FILE = "../models/movement_labels.json"

# Số frame cố định của mỗi sequence sau khi chuẩn hóa
SEQ_LEN = 100    # Bạn có thể đổi 30–60 tùy bài toán


def load_sequences():
    X = []
    y = []
    labels = []

    # Duyệt qua từng folder movement
    for label in sorted(os.listdir(RAW_DIR)):
        class_dir = os.path.join(RAW_DIR, label)
        if not os.path.isdir(class_dir):
            continue

        labels.append(label)
        class_index = labels.index(label)

        for file in os.listdir(class_dir):
            if not file.endswith(".json"):
                continue

            filepath = os.path.join(class_dir, file)

            with open(filepath, "r") as f:
                data = json.load(f)

            seq = np.array(data["sequence"], dtype=np.float32)  # shape: (frames, 63)

            # Bỏ qua sequence quá ngắn (<5 frames)
            if seq.shape[0] < 5:
                continue

            # Chuẩn hóa độ dài
            seq_fixed = normalize_sequence(seq, SEQ_LEN)

            X.append(seq_fixed)
            y.append(class_index)

    return np.array(X), np.array(y), labels


def normalize_sequence(seq, target_len):
    """
    Chuẩn hóa độ dài sequence thành target_len bằng nội suy (interpolation)
    """
    original_len = seq.shape[0]

    # Tạo index mới đều nhau
    new_idx = np.linspace(0, original_len - 1, target_len)
    new_seq = []

    for i in range(target_len):
        frame = seq[int(new_idx[i])]  # Lấy frame gần nhất
        new_seq.append(frame)

    return np.array(new_seq, dtype=np.float32)


def save_labels(labels):
    with open(LABEL_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print("Đã lưu labels vào:", LABEL_FILE)


def main():
    print("Đang load dữ liệu movement...")
    X, y, labels = load_sequences()

    print("Shape X:", X.shape)   # (samples, SEQ_LEN, 63)
    print("Shape y:", y.shape)

    save_labels(labels)

    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)

    print("Đã tạo dataset movement thành công:")
    print(" →", OUTPUT_X)
    print(" →", OUTPUT_Y)
    print(" →", LABEL_FILE)


if __name__ == "__main__":
    main()
