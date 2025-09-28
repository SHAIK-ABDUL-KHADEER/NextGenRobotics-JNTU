import cv2
import mediapipe as mp
import time

# Try to import serial
try:
    import serial
    arduino = serial.Serial('COM7', 9600, timeout=1)
    time.sleep(2)
    led_enabled = True
    print("Arduino connected, LED control enabled")
except:
    arduino = None
    led_enabled = False
    print("Arduino not connected, running without LED")

# Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def hand_status(hand_landmarks):
    fingers = [8, 12, 16, 20]
    folded = 0
    for tip in fingers:
        if hand_landmarks.landmark[tip].y > hand_landmarks.landmark[tip - 2].y:
            folded += 1
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        folded += 0
    else:
        folded += 1
    return "Closed" if folded >= 4 else "Open"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    status_val = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            status_val = hand_status(hand_landmarks)
            print("Hand:", status_val)

            if led_enabled:
                if status_val == "Open":
                    arduino.write(b"LED_ON\n")
                else:
                    arduino.write(b"LED_OFF\n")

    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
if led_enabled:
    arduino.close()
