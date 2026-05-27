#!/usr/bin/env python3
"""
tools/train_movement.py

Train a lightweight temporal model (GRU or TCN) on movement dataset produced
by tools/build_movement_dataset.py.

Outputs:
 - models/movement_model.h5
 - models/movement_model.tflite
 - models/movement_labels.json (reused)
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------- CONFIG ----------------
SEQ_LEN = 100            # must match dataset builder
INPUT_DIM = 63
BATCH_SIZE = 32
EPOCHS = 80
PATIENCE = 8
MODEL_DIR = "../models"
H5_OUT = os.path.join(MODEL_DIR, "movement_model.h5")
TFLITE_OUT = os.path.join(MODEL_DIR, "movement_model.tflite")
# ----------------------------------------

def load_data():
    X = np.load("../data/movement_X.npy", allow_pickle=False)
    y = np.load("../data/movement_y.npy", allow_pickle=False)
    # shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    return X, y

def build_gru_model(seq_len=SEQ_LEN, input_dim=INPUT_DIM, num_classes=10, dropout=0.2):
    inp = keras.Input(shape=(seq_len, input_dim), name="input_landmarks")
    x = layers.Masking(mask_value=0.0)(inp)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.GRU(64)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax", name="preds")(x)
    model = keras.Model(inp, out, name="gru_movement")
    return model

def build_tcn_model(seq_len=SEQ_LEN, input_dim=INPUT_DIM, num_classes=10, dropout=0.2):
    # Lightweight TCN implementation using Conv1D + dilation
    inputs = keras.Input(shape=(seq_len, input_dim))
    x = layers.Masking(mask_value=0.0)(inputs)
    def tcn_block(x, filters, kernel_size, dilation_rate):
        prev = x
        x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        # residual
        if prev.shape[-1] != filters:
            prev = layers.Conv1D(filters, 1, padding="same")(prev)
        x = layers.Add()([prev, x])
        x = layers.Activation("relu")(x)
        return x

    x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
    for d in [1, 2, 4]:
        x = tcn_block(x, 64, 3, dilation_rate=d)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="tcn_movement")
    return model

def representative_dataset_gen(X, num_samples=100):
    # yields float32 input samples for quantization
    def gen():
        n = min(len(X), num_samples)
        for i in range(n):
            sample = X[i].astype(np.float32)
            sample = np.expand_dims(sample, axis=0)  # shape (1, seq_len, 63)
            yield [sample]
    return gen

def convert_to_tflite(keras_model, tflite_path=TFLITE_OUT):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    # Cho phép ops TensorFlow (GRU/LSTM)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    # Tắt hạ cấp TensorList (quan trọng)
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print("Saved TFLite (with SELECT_TF_OPS) to:", tflite_path)


def main(args):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    X, y = load_data()
    print("Dataset shapes:", X.shape, y.shape)

    num_classes = int(np.max(y) + 1)
    print("Num classes:", num_classes)

    # train/val split (subject-wise preferred; here simple split)
    val_frac = args.val_frac
    n = len(X)
    split = int(n * (1 - val_frac))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # one-hot
    y_train_o = keras.utils.to_categorical(y_train, num_classes)
    y_val_o = keras.utils.to_categorical(y_val, num_classes)

    # build model
    if args.model_type.lower() == "gru":
        model = build_gru_model(seq_len=SEQ_LEN, input_dim=INPUT_DIM, num_classes=num_classes, dropout=args.dropout)
    else:
        model = build_tcn_model(seq_len=SEQ_LEN, input_dim=INPUT_DIM, num_classes=num_classes, dropout=args.dropout)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(H5_OUT, save_best_only=True, monitor="val_accuracy", mode="max"),
        keras.callbacks.EarlyStopping(patience=PATIENCE, monitor="val_loss", restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
    ]

    # class weights (optional) to handle imbalance
    classes, counts = np.unique(y_train, return_counts=True)
    total = counts.sum()
    class_weights = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    print("Class weights:", class_weights)

    print("Start training...")
    history = model.fit(
        X_train, y_train_o,
        validation_data=(X_val, y_val_o),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=2
    )

    # save final keras model (best already saved by checkpoint)
    if not os.path.exists(H5_OUT):
        model.save(H5_OUT)
    print("Saved Keras model to:", H5_OUT)

    # Convert to TFLite
    if args.quantize:
        print("Converting to quantized TFLite (INT8) using representative dataset...")
        # use a subset of training data as representative
        convert_to_tflite(model, tflite_path=TFLITE_OUT)
    else:
        print("Converting to TFLite (float32)...")
        convert_to_tflite(keras.models.load_model(H5_OUT), quantize=False, tflite_path=TFLITE_OUT)

    # save class labels (if not exist)
    labels_path = os.path.join(MODEL_DIR, "movement_labels.json")
    if os.path.exists("models/movement_labels.json"):  # if builder already saved
        pass
    else:
        # create labels simple 0..N-1
        labels = [f"class_{i}" for i in range(num_classes)]
        with open(labels_path, "w", encoding="utf8") as f:
            json.dump(labels, f, indent=2)
        print("Saved labels to", labels_path)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="gru", help="gru or tcn")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--quantize", action="store_true", help="Export INT8 quantized TFLite")
    args = parser.parse_args()
    # update globals from cli
    SEQ_LEN = SEQ_LEN
    BATCH_SIZE = args.batch
    EPOCHS = args.epochs
    main(args)
