import cv2
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

# Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) > 0 and led_enabled:
        arduino.write(b"LED_ON\n")
    elif led_enabled:
        arduino.write(b"LED_OFF\n")

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
if led_enabled:
    arduino.close()
