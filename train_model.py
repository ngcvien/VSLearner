#!/usr/bin/env python3
# train_model.py
# Huấn luyện model từ thư mục data/<label>/*.csv
# Mỗi dòng CSV được coi là một sample (63 features: 21 landmarks * 3)

import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import layers, callbacks, models
import pickle

DATA_DIR = "data"   
MODEL_OUT = "sign_model.h5"
SCALER_OUT = "scaler.pkl"
LABELS_OUT = "labels.json"
RANDOM_SEED = 42

# 1. Đọc dữ liệu
def load_data(data_dir=DATA_DIR):
    X_list = []
    y_list = []
    labels = []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        labels.append(label)
        csv_files = glob.glob(os.path.join(label_dir, "*.csv"))
        for f in csv_files:
            try:
                df = pd.read_csv(f, header=0)  # nếu header=0 (do pandas lưu)
            except Exception:
                df = pd.read_csv(f, header=None)
            # Nếu file rỗng thì skip
            if df.shape[0] == 0:
                continue
            # Nếu có nhiều cột >63 (do vô tình lưu nhãn) -> take first 63
            if df.shape[1] >= 63:
                arr = df.values[:, :63]
            else:
                # nếu thiếu cột, pad bằng 0 (ít khả năng nhưng an toàn)
                pad = np.zeros((df.shape[0], 63 - df.shape[1]))
                arr = np.hstack([df.values, pad])
            X_list.append(arr.astype(np.float32))
            y_list += [label] * arr.shape[0]

    if len(X_list) == 0:
        raise RuntimeError(f"Không tìm thấy dữ liệu trong {data_dir}. Hãy chắc rằng bạn có thư mục label chứa csv.")
    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y, labels

print("Đang load dữ liệu...")
X, y, labels_found = load_data()
print(f"→ Tổng mẫu: {X.shape[0]}, features mỗi mẫu: {X.shape[1]}")
print(f"→ Nhãn phát hiện: {labels_found}")

# 2. Mã hoá nhãn
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"→ Số lớp: {num_classes} -> {list(le.classes_)}")

# 3. Shuffle + split
X, y_enc = shuffle(X, y_enc, random_state=RANDOM_SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=RANDOM_SEED, stratify=y_enc)

# 4. Chuẩn hoá (StandardScaler)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lưu scaler và labels
with open(SCALER_OUT, "wb") as f:
    pickle.dump(scaler, f)
with open(LABELS_OUT, "w", encoding="utf-8") as f:
    json.dump({"classes": le.classes_.tolist()}, f, ensure_ascii=False, indent=2)
print(f"✅ Đã lưu scaler -> {SCALER_OUT} và labels -> {LABELS_OUT}")

# 5. Tạo model MLP đơn giản
def build_mlp(input_dim, num_classes):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

model = build_mlp(X_train.shape[1], num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# Callbacks
es = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
mc = callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True)

# 6. Huấn luyện
history = model.fit(X_train, y_train,
                    validation_split=0.12,
                    epochs=100,
                    batch_size=64,
                    callbacks=[es, mc],
                    verbose=2)

# 7. Đánh giá trên test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🔎 Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

# 8. Lưu model (nếu ModelCheckpoint đã lưu rồi thì optional)
if not os.path.exists(MODEL_OUT):
    model.save(MODEL_OUT)
print(f"✅ Mô hình đã lưu tại: {MODEL_OUT}")

# 9. In chi tiết
from sklearn.metrics import classification_report, confusion_matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
