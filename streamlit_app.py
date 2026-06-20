import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
from math import dist
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("🚗 Driver Drowsiness Detection")
st.write("Real-time eye closure detection - Works in browser!")

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

# Initialize session state
if "counter" not in st.session_state:
    st.session_state.counter = 0
if "alert_count" not in st.session_state:
    st.session_state.alert_count = 0

# WebRTC configuration
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.sidebar.header("⚙️ Settings")
ear_threshold = st.sidebar.slider("EAR Threshold", 0.1, 0.5, 0.22, 0.01)
consec_frames = st.sidebar.slider("Alert Frames", 5, 50, 20, 1)

class DrowsinessProcessor:
    def __init__(self):
        self.counter = 0
        self.alerts = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, c = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = []
                right_eye = []

                for idx in LEFT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    left_eye.append((x, y))
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

                for idx in RIGHT_EYE:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    right_eye.append((x, y))
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2

                cv2.putText(
                    img,
                    f"EAR: {avg_ear:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2
                )

                if avg_ear < ear_threshold:
                    self.counter += 1
                else:
                    self.counter = 0

                if self.counter >= consec_frames:
                    cv2.putText(
                        img,
                        "DROWSINESS ALERT!",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )
                    self.alerts += 1
        else:
            cv2.putText(
                img,
                "Face not detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (100, 100, 100),
                2
            )
            self.counter = 0

        return img

# WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"audio": False, "video": True},
    async_processing=True,
    video_processor_factory=DrowsinessProcessor,
)

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Status", "🟢 Running" if webrtc_ctx.state.playing else "🔴 Stopped")
with col2:
    st.info("Allow camera access when prompted")
with col3:
    st.write("")

st.markdown("""
### How it works:
- **Eye Aspect Ratio (EAR):** Measures eye openness
- **Threshold:** Eyes closed when EAR < 0.22
- **Alert:** Triggers after 20+ frames of closed eyes
""")