# inference_service_tflite.py
# Flask service sử dụng TFLite model để predict.
# Hỗ trợ 2 kiểu input: "keypoints" (list length 63) hoặc "image" (base64 jpeg).
# Tự động xử lý quantized model (int8/uint8) nếu cần.

import os, time, json, base64, io
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
from PIL import Image
import logging

# TFLite: dùng tflite_runtime hoặc tensorflow lite
try:
    # prefer tflite_runtime on Pi
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
except Exception:
    # fallback to TF
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# Optional scaler for keypoints (if you exported scaler.pkl from training)
try:
    import pickle
    SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
    scaler = pickle.load(open(SCALER_PATH, 'rb')) if os.path.exists(SCALER_PATH) else None
except Exception:
    scaler = None

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sign_model.tflite')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')

# Flask init
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
logging.basicConfig(level=logging.INFO)

# global interpreter holder
_interpreter = None
_input_details = None
_output_details = None
_labels = []

def load_labels():
    global _labels
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r', encoding='utf8') as f:
            j = json.load(f)
            _labels = j.get('classes', [])
    else:
        _labels = []
    logging.info(f"Labels loaded: {_labels}")

def load_model(path=MODEL_PATH):
    global _interpreter, _input_details, _output_details
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    # load interpreter
    logging.info("Loading TFLite model: " + path)
    _interpreter = Interpreter(model_path=path)
    _interpreter.allocate_tensors()
    _input_details = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()
    logging.info("Model input_details: %s", _input_details)
    logging.info("Model output_details: %s", _output_details)

def preprocess_image_b64(b64str, target_shape):
    # b64str like 'data:image/jpeg;base64,...' or raw base64
    if b64str.startswith('data:'):
        b64str = b64str.split(',',1)[1]
    im = Image.open(io.BytesIO(base64.b64decode(b64str))).convert('RGB')
    im = im.resize((target_shape[1], target_shape[2]))  # (N,H,W,C) expected for input_details maybe
    arr = np.asarray(im).astype(np.float32) / 255.0
    # shape to (1,H,W,3)
    if len(arr.shape) == 3:
        arr = np.expand_dims(arr, 0)
    return arr

def preprocess_keypoints(kp_list, expected_dim):
    # kp_list: list of floats length 63
    x = np.array(kp_list, dtype=np.float32).reshape(1, -1)
    if scaler is not None:
        try:
            x = scaler.transform(x)
        except Exception as e:
            logging.warning("Scaler transform failed: %s", e)
    # if model input expects shape (1,63) OK, else try reshape
    if expected_dim == x.shape[1]:
        return x
    # if model input expects (1, N) with N matching, else attempt pad/trim
    # expected_dim here is input_details shape product
    return x

def set_input_tensor_from_array(arr):
    # arr: numpy array shaped as interpreter input expects, dtype handled below
    idx = _input_details[0]['index']
    dtype = _input_details[0]['dtype']
    # handle quantization
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = _input_details[0].get('quantization', (1.0, 0))
        if scale and zero_point is not None:
            # map float arr in [0,1] or arbitrary to int8 using scale
            arr_int = np.round(arr / scale + zero_point).astype(dtype)
        else:
            arr_int = arr.astype(dtype)
        _interpreter.set_tensor(idx, arr_int)
    else:
        _interpreter.set_tensor(idx, arr.astype(dtype))

def predict_from_keypoints(kp_list):
    # try to map to model input:
    # determine input shape
    inp_shape = _input_details[0]['shape']  # e.g., [1,63] or [1,64,64,3]
    inp_dim = int(np.prod(inp_shape[1:])) if len(inp_shape)>1 else inp_shape[0]
    # if shape is 2D and second dim == 63 -> use directly
    if len(inp_shape) == 2 and inp_shape[1] in (len(kp_list), 63):
        arr = preprocess_keypoints(kp_list, inp_shape[1])
        set_input_tensor_from_array(arr)
    else:
        # fallback: try to make a small image from keypoints (not ideal)
        # create flat vector padded/truncated to expected size
        flat = np.array(kp_list, dtype=np.float32).reshape(1,-1)
        target = np.zeros((1, inp_dim), dtype=np.float32)
        copy_len = min(flat.size, target.size)
        target.flat[:copy_len] = flat.flat[:copy_len]
        set_input_tensor_from_array(target)
    _interpreter.invoke()
    out = _interpreter.get_tensor(_output_details[0]['index'])
    return out

def predict_from_image_b64(b64str):
    inp_shape = _input_details[0]['shape']  # e.g., [1,H,W,3]
    if len(inp_shape) == 4:
        arr = preprocess_image_b64(b64str, inp_shape)
        set_input_tensor_from_array(arr)
    elif len(inp_shape) == 2:
        # model expects flat vector; we can extract simple features (e.g., mean color)
        arr_img = preprocess_image_b64(b64str, (1, inp_shape[1], 1))
        arr_flat = arr_img.reshape(1, -1)[:,:_input_details[0]['shape'][1]]
        set_input_tensor_from_array(arr_flat)
    else:
        raise RuntimeError("Unsupported model input shape: "+str(inp_shape))
    _interpreter.invoke()
    out = _interpreter.get_tensor(_output_details[0]['index'])
    return out

def postprocess_output(out):
    # out: numpy array (1, num_classes) or logits
    probs = np.squeeze(out)
    if probs.ndim == 0:
        # regression? fallback
        return {"label":"unknown", "confidence": float(probs)}
    # if quantized output, dequantize
    if _output_details[0]['dtype'] in (np.uint8, np.int8):
        scale, zero_point = _output_details[0].get('quantization', (1.0,0))
        probs = scale * (probs.astype(np.float32) - zero_point)
        # if not softmaxed, might be logits; apply softmax
    # if sum not ~1, apply softmax
    if probs.sum() <= 0 or probs.sum() > 1.0001:
        # softmax
        e = np.exp(probs - np.max(probs))
        probs = e / e.sum()
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = _labels[idx] if idx < len(_labels) else str(idx)
    return {"label": label, "confidence": round(conf, 3)}

# Load model + labels on startup
load_labels()
load_model(MODEL_PATH)

# REST endpoints
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    typ = data.get('type', 'image')
    try:
        if typ == 'keypoints':
            kp = data.get('keypoints', [])
            out = predict_from_keypoints(kp)
        else:
            img_b64 = data.get('image_b64', '')
            out = predict_from_image_b64(img_b64)
        res = postprocess_output(out)
        res['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        return jsonify(res)
    except Exception as e:
        logging.exception("Predict error")
        return jsonify({"error": str(e)}), 500

@app.route('/reload_model', methods=['POST'])
def reload_model_route():
    # optional: accept JSON {"model_path": "..."} to hot-reload
    j = request.get_json(force=True)
    p = j.get('model_path', MODEL_PATH)
    try:
        load_model(p)
        return jsonify({"status":"ok", "model": p})
    except Exception as e:
        return jsonify({"status":"error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status":"ok", "model_loaded": True if _interpreter else False})

@app.route('/enroll', methods=['POST'])
def enroll():
    j = request.get_json(force=True)
    user = j.get('user_id','unknown'); label = j.get('label','unknown')
    samples = j.get('samples', [])
    outdir = os.path.join(BASE_DIR, 'data', 'enroll', user, label)
    os.makedirs(outdir, exist_ok=True)
    for i,s in enumerate(samples):
        fname = os.path.join(outdir, f'sample_{int(time.time())}_{i}.json')
        with open(fname, 'w', encoding='utf8') as f:
            json.dump(s, f, ensure_ascii=False)
    return jsonify({"status":"ok","saved":len(samples)})

@socketio.on('connect')
def on_connect():
    emit('status', {'msg':'connected'})

if __name__ == '__main__':
    logging.info("Starting TFLite inference service on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000)
