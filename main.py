import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time

# ---------- SCREEN ----------
screen_w, screen_h = pyautogui.size()

# ---------- MODES ----------
mode = "mouse"

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

mp_draw = mp.solutions.drawing_utils

# ---------- DRAWING ----------
canvas = np.zeros((480, 640, 3), np.uint8)
prev_x, prev_y = 0, 0

# ---------- STUDY METRICS ----------
blink_frames = 0
blink_counter = 0

yawn_counter = 0
yawn_start_time = None

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# ---------- FUNCTIONS ----------
def fingers_up(handLms):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if handLms.landmark[4].x < handLms.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for tip in tips[1:]:
        if handLms.landmark[tip].y < handLms.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# ---------- MAIN LOOP ----------
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # -------- HAND DETECTION --------
    hand_results = hands.process(img_rgb)

    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            fingers = fingers_up(handLms)

            index = handLms.landmark[8]
            cx, cy = int(index.x * w), int(index.y * h)

            # ----- MOUSE MODE -----
            if mode == "mouse":
                screen_x = np.interp(cx, [0, w], [0, screen_w])
                screen_y = np.interp(cy, [0, h], [0, screen_h])
                pyautogui.moveTo(screen_x, screen_y)

                thumb = handLms.landmark[4]
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                dist = math.hypot(cx - tx, cy - ty)

                if dist < 35:
                    pyautogui.click()

            # ----- DRAW MODE -----
            if mode == "draw":
                if fingers == [0, 1, 0, 0, 0]:  # index finger only
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = cx, cy
                    cv2.line(canvas, (prev_x, prev_y), (cx, cy), (255, 0, 0), 5)
                    prev_x, prev_y = cx, cy

                elif fingers == [1, 1, 1, 1, 1]:  # clear canvas
                    canvas = np.zeros((480, 640, 3), np.uint8)

                else:
                    prev_x, prev_y = 0, 0

    # -------- FACE DETECTION (STUDY MODE) --------
    face_results = face_mesh.process(img_rgb)

    if face_results.multi_face_landmarks and mode == "study":
        for face in face_results.multi_face_landmarks:

            # ----- BLINK DETECTION -----
            eye_top = face.landmark[159]
            eye_bottom = face.landmark[145]
            eye_dist = abs(eye_top.y - eye_bottom.y)

            if eye_dist < 0.02:
                blink_frames += 1
            else:
                if blink_frames > 2:
                    blink_counter += 1
                blink_frames = 0

            # ----- YAWN TIME DETECTION -----
            mouth_top = face.landmark[13]
            mouth_bottom = face.landmark[14]
            mouth_dist = abs(mouth_top.y - mouth_bottom.y)

            current_time = time.time()

            if mouth_dist > 0.06:
                if yawn_start_time is None:
                    yawn_start_time = current_time
            else:
                if yawn_start_time is not None:
                    duration = current_time - yawn_start_time
                    if duration > 1.5:
                        yawn_counter += 1
                    yawn_start_time = None

            # Debug Text
            cv2.putText(img, f"EyeDist: {round(eye_dist,3)}", (20,160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.putText(img, f"MouthDist: {round(mouth_dist,3)}", (20,190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # -------- OVERLAY CANVAS --------
    img = cv2.addWeighted(img, 1, canvas, 1, 0)

    # -------- UI --------
    cv2.putText(img, f"Mode: {mode}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if mode == "study":
        cv2.putText(img, f"Blinks: {blink_counter}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
        cv2.putText(img, f"Yawns: {yawn_counter}", (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("AI Vision Suite", img)

    # -------- KEY CONTROLS --------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = "mouse"
    elif key == ord('d'):
        mode = "draw"
    elif key == ord('s'):
        mode = "study"
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






