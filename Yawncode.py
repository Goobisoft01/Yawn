import cv2 
import mediapipe as mp
import time

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

# Variables to track yawn duration
yawn_start_time = None
yawn_threshold = 1.5 # Seconds the mouth must remain open to trigger alarm

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with mediapipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face landmarks
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Extract mouth landmarks
            mouth_top = face_landmarks.landmark[13]  # Upper lip
            mouth_bottom = face_landmarks.landmark[14]  # Lower lip

            # Calculate mouth openness
            mouth_openness = abs(mouth_top.y - mouth_bottom.y)

            # Set threshold for mouth openness
            if mouth_openness > 0.04:  # Adjust based on calibration
                if yawn_start_time is None:
                    yawn_start_time = time.time()  # Start the timer
                else:
                    elapsed_time = time.time() - yawn_start_time
                    if elapsed_time > yawn_threshold:
                        cv2.putText(frame, "Yawning Detected!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                yawn_start_time = None  # Reset the timer if the mouth closes

    # Show the frame
    cv2.imshow('Mouth Detection', frame)

    # Exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
