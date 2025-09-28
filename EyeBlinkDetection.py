import cv2
import mediapipe as mp
import time

# ----------------- Serial Setup (Optional) -----------------
try:
    import serial
    arduino = serial.Serial('COM7', 9600, timeout=1)  # Replace with your Arduino port
    time.sleep(2)
    led_enabled = True
    print("Arduino connected, LED control enabled")
except:
    arduino = None
    led_enabled = False
    print("Arduino not connected, running without LED")

# ----------------- MediaPipe Face Mesh -----------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.25
prev_time = time.time()

# ----------------- Function to Calculate EAR -----------------
def eye_aspect_ratio(landmarks, eye_indices):
    p = [landmarks[i] for i in eye_indices]
    vert1 = ((p[1].x - p[5].x)**2 + (p[1].y - p[5].y)**2) ** 0.5
    vert2 = ((p[2].x - p[4].x)**2 + (p[2].y - p[4].y)**2) ** 0.5
    horz = ((p[0].x - p[3].x)**2 + (p[0].y - p[3].y)**2) ** 0.5
    return (vert1 + vert2) / (2 * horz)

# ----------------- Webcam Capture -----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    eye_status = None
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Draw Face Mesh
        mp_drawing.draw_landmarks(
            frame,
            results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
        )

        # Calculate EAR
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2

        # Print status every 0.1 sec
        if time.time() - prev_time >= 0.1:
            prev_time = time.time()
            eye_status = "Closed" if avg_ear < EAR_THRESHOLD else "Open"
            print("Eye:", eye_status)

            # LED control
            if led_enabled:
                if eye_status == "Open":
                    arduino.write(b"LED_ON\n")
                else:
                    arduino.write(b"LED_OFF\n")

    cv2.imshow("Eye Blink Detection with Face Mesh", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
if led_enabled:
    arduino.close()
