# camera_service_keypoints.py
import cv2, mediapipe as mp, requests, time, json

SERVER = "http://localhost:5000/predict"  # đổi nếu server khác
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

print("Open camera. Press q to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    display = frame.copy()
    kp = None
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(display, lm, mp_hands.HAND_CONNECTIONS)
        # build keypoints array 63
        kp = []
        for p in lm.landmark:
            kp += [p.x, p.y, p.z]
        # send to server
        try:
            r = requests.post(SERVER, json={'type':'keypoints', 'keypoints': kp}, timeout=1.0)
            if r.status_code == 200:
                j = r.json()
                label = j.get('label','-')
                conf = j.get('confidence', j.get('conf',0))
                cv2.putText(display, f"{label} ({conf})", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
                print("Pred:", j)
            else:
                print("Server returned", r.status_code, r.text)
        except Exception as e:
            print("Req error:", e)
    else:
        cv2.putText(display, "No hand", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    cv2.imshow("Cam Keypoints -> Server", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
