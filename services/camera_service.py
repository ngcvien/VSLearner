# camera_service.py
# Mô tả: mở webcam, crop nhỏ, gửi POST /predict (image base64)
import cv2, base64, requests, time


SERVER = 'http://localhost:5000/predict' # đổi thành địa chỉ server nếu cần
CAPTURE_FPS = 5


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Cannot open camera')
    exit()


print('Camera opened. Sending frames to', SERVER)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # resize để giảm bandwidth
        roi = cv2.resize(frame, (160, 120))
        _, buf = cv2.imencode('.jpg', roi, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        b64 = base64.b64encode(buf.tobytes()).decode('ascii')
        payload = { 'type':'image', 'image_b64': 'data:image/jpeg;base64,' + b64 }
        try:
            r = requests.post(SERVER, json=payload, timeout=1.0)
            print('Resp:', r.json())
        except Exception as e:
            print('Err:', e)
        cv2.imshow('Cam', roi)
        if cv2.waitKey(int(1000/CAPTURE_FPS)) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()