import cv2, mediapipe as mp
cap = cv2.VideoCapture(0)
mph = mp.solutions.holistic
md = mp.solutions.drawing_utils

# Cấu hình vẽ
dot_spec  = md.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=3)  # chấm
line_spec = md.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=0)  # đường

with mph.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as h:
    while True:
        ret, img = cap.read()
        if not ret: break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        r = h.process(img_rgb)

        if r.pose_landmarks:
            md.draw_landmarks(img, r.pose_landmarks, mph.POSE_CONNECTIONS,
                              landmark_drawing_spec=dot_spec,
                              connection_drawing_spec=line_spec)
        if r.left_hand_landmarks:
            md.draw_landmarks(img, r.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                              landmark_drawing_spec=dot_spec,
                              connection_drawing_spec=line_spec)
        if r.right_hand_landmarks:
            md.draw_landmarks(img, r.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                              landmark_drawing_spec=dot_spec,
                              connection_drawing_spec=line_spec)

        cv2.imshow('Mediapipe Skeleton', img)
        if cv2.waitKey(1) & 0xFF == 27: break  # ESC thoát

cap.release()
cv2.destroyAllWindows()
