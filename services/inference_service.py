# inference_service.py
# Flask + SocketIO stub server
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import time, random, json, os


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


# load labels nếu có
LABELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'labels.json')
try:
    with open(LABELS_PATH, 'r', encoding='utf8') as f:
        labels = json.load(f).get('classes', ['A','B','Hello'])
except Exception:
    labels = ['A','B','Hello']


print('Loaded labels:', labels)


# stub predict từ keypoints hoặc image
def predict_stub_from_keypoints(kp):
# kp: list floats
    if not kp:
        return {"label": random.choice(labels), "confidence": round(random.random(),2)}
    s = int(sum(kp) * 1000) if isinstance(kp, list) else random.randint(0,999)
    idx = s % len(labels)
    conf = 0.6 + (s % 40)/100.0
    return {"label": labels[idx], "confidence": round(conf,2)}


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    typ = data.get('type')
    if typ == 'keypoints':
        kp = data.get('keypoints', [])
        res = predict_stub_from_keypoints(kp)
    else:
# image or other
        res = {"label": random.choice(labels), "confidence": round(random.random(),2)}
    res['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    return jsonify(res)


@app.route('/enroll', methods=['POST'])
def enroll():
    j = request.get_json(force=True)
    user = j.get('user_id','unknown')
    label = j.get('label','unknown')
    samples = j.get('samples', [])
    # lưu samples đơn giản vào data/enroll/<user>/<label>/
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'enroll', user)
    os.makedirs(base, exist_ok=True)
    out_dir = os.path.join(base, label)
    os.makedirs(out_dir, exist_ok=True)
    # mỗi samples là 1 object -> lưu thành file json
    for i, s in enumerate(samples):
        fp = os.path.join(out_dir, f'sample_{int(time.time())}_{i}.json')
        with open(fp, 'w', encoding='utf8') as f:
            json.dump(s, f, ensure_ascii=False)
    return jsonify({"status":"ok","message":"enrolled","prototype_id":f"{user}_{label}"})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status":"ok","model_loaded":False})


@socketio.on('connect')
def on_connect():
    print('Client connected')
    emit('status', {'msg':'connected'})


if __name__ == '__main__':
    print('Starting inference_service stub on http://0.0.0.0:5000')
    socketio.run(app, host='0.0.0.0', port=5000)