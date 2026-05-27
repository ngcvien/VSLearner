import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

prev = None
SMOOTH = 0.8
energy = 0.0

START_TH = 0.008
STOP_TH  = 0.004

state = "IDLE"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        cur = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()

        if prev is not None:
            diff = np.mean(np.abs(cur - prev))
            energy = SMOOTH * energy + (1 - SMOOTH) * diff

        prev = cur

        if state == "IDLE" and energy > START_TH:
            state = "MOVING"
            print("[START]")

        if state == "MOVING" and energy < STOP_TH:
            state = "IDLE"
            print("[STOP]")

        cv2.putText(img, f"Energy: {energy:.4f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(img, f"State: {state}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Movement Energy Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
