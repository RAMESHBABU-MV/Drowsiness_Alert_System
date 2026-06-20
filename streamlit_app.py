import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
from math import dist

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("Driver Drowsiness Detection")
st.write("This application uses your webcam to detect drowsiness based on your Eye Aspect Ratio (EAR).")

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Drowsiness settings
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 20

def eye_aspect_ratio(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical1 = dist(p2, p6)
    vertical2 = dist(p3, p5)
    horizontal = dist(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

if "counter" not in st.session_state:
    st.session_state.counter = 0

if run:
    cap = cv2.VideoCapture(0)
    
    while run:
        success, frame = cap.read()
        
        if not success:
            st.error("Failed to read from webcam. Please check your camera permissions.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                left_eye = []
                right_eye = []

                for idx in LEFT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    left_eye.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                for idx in RIGHT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    right_eye.append((x, y))
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)

                avg_ear = (left_ear + right_ear) / 2

                cv2.putText(
                    frame,
                    f"EAR: {avg_ear:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2
                )

                if avg_ear < EAR_THRESHOLD:
                    st.session_state.counter += 1
                else:
                    st.session_state.counter = 0

                if st.session_state.counter >= CONSEC_FRAMES:
                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )

        # Convert the BGR frame back to RGB for Streamlit rendering
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
    
    cap.release()
else:
    st.write("Webcam is stopped.")
