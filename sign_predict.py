import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import json
import pickle
import os

# --- Paths 
MODEL_PATH = "sign_model.h5"
SCALER_PATH = "scaler.pkl"
LABELS_PATH = "labels.json"

# --- Load model, scaler, labels ---
model = tf.keras.models.load_model(MODEL_PATH)

scaler = None
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf8") as f:
        labels = json.load(f).get("classes", [])
else:
    # fallback: nếu bạn chưa có labels.json, hãy sửa list này cho khớp
    labels = ["A", "B", "Hello"]
    print("⚠️ labels.json không tìm thấy, dùng fallback labels (hãy tạo labels.json từ train).")

print(f"Loaded model. Number of labels = {len(labels)}")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

engine = pyttsx3.init()
last_label = None

cap = cv2.VideoCapture(0)
print("Đang chạy inference... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    display_text = "No hand"
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0]
        coords = []
        for p in lm.landmark:
            coords += [p.x, p.y, p.z]
        x = np.array(coords, dtype=np.float32).reshape(1, -1)  # shape (1,63) expected

        # apply scaler nếu có
        if scaler is not None:
            try:
                x = scaler.transform(x)
            except Exception as e:
                print("⚠️ Lỗi khi scaler.transform:", e)

        try:
            pred = model.predict(x, verbose=0)  # shape (1, num_classes)
            # đảm bảo pred là numpy array
            pred = np.asarray(pred)
            if pred.ndim == 2:
                probs = pred[0]
            elif pred.ndim == 1:
                probs = pred
            else:
                # unexpected shape
                print("⚠️ Prediction có shape bất thường:", pred.shape)
                probs = pred.flatten()

            pred_idx = int(np.argmax(probs))
            # Debug: kiểm tra kích thước và index
            if pred_idx >= len(labels):
                print("⚠️ Index dự đoán vượt quá labels length!")
                print(" prediction shape:", pred.shape, " argmax:", pred_idx, " labels_len:", len(labels))
                # show top-k để debug
                topk = np.argsort(probs)[-5:][::-1]
                print(" Top probs indexes:", topk, "values:", probs[topk])
                display_text = "Unknown"
            else:
                label = labels[pred_idx]
                conf = probs[pred_idx]
                if conf < 0.9:
                    display_text = "Unknown"
                else:
                    display_text = f"{label} ({conf:.2f})"
                # đọc âm thanh khi label thay đổi
                if label != last_label:
                    engine.say(label)
                    engine.runAndWait()
                    last_label = label

        except Exception as e:
            print("⚠️ Lỗi khi predict:", e)
            display_text = "Error"

        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, display_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Sign Predict (safe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
