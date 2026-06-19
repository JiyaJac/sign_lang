import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ── Load your model — identical to execution.py ────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ── MediaPipe setup — identical to execution.py ────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ── WebRTC connection config ────────────────────────────────────────────────
# Explicit STUN server — without this, some networks (corporate firewalls,
# certain routers/VPNs, some Windows configs) fail to negotiate the
# peer-to-peer camera connection, causing it to hang on "Connection is
# taking longer than expected."
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sign Language Detector", page_icon="✋", layout="wide")
st.title("✋ Sign Language Detector")
st.caption("Show a sign, hold it steady — it appears below.")

# ── Persistent state across reruns (Streamlit reruns the script often) ────
# st.session_state survives between frames/interactions, unlike normal variables
if "word" not in st.session_state:
    st.session_state.word = ""
if "current_letter" not in st.session_state:
    st.session_state.current_letter = ""

# ── Video processor — this class runs once per frame, same role as your ──
# while-loop body in execution.py. It keeps its OWN state (prev_prediction,
# count, inserted) because it lives inside the streaming thread, separate
# from Streamlit's normal script reruns.
class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.hand = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.prev_prediction = ""
        self.count = 0
        self.inserted = False
        self.current_letter = ""
        self.word = ""
        self.sign_label = ""   # what to show as the live "Sign:" badge

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)                         # mirror, same as execution.py
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hand.process(rgb)

        data = []
        prediction = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])

            if len(data) == 63:
                prediction = model.predict([data])[0]

                if prediction == self.prev_prediction:
                    self.count += 1
                else:
                    self.count = 0
                    self.inserted = False

                if self.count > 10 and not self.inserted:
                    if prediction == "space":
                        self.word += " "
                        self.current_letter = " "
                        self.inserted = True
                    elif prediction == "del":
                        if self.current_letter != "":
                            self.current_letter = ""
                        else:
                            self.word = self.word[:-1]
                        self.inserted = True
                    elif prediction == "next":
                        self.word += self.current_letter
                        self.current_letter = ""
                        self.inserted = True
                    else:
                        self.current_letter = prediction
                        self.inserted = True

                self.prev_prediction = prediction
                self.sign_label = f"{prediction} ({'stable' if self.count > 10 else 'detecting'})"
        else:
            self.sign_label = "No hand detected"

        # Draw overlay text directly on the frame — same as your cv2.putText calls
        cv2.putText(img, f"Sign: {self.sign_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Letter: {self.current_letter}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.putText(img, f"Word: {self.word}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── Start the camera widget ──────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    ctx = webrtc_streamer(
        key="sign-detector",
        video_processor_factory=SignProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Live Output")
    placeholder_letter = st.empty()
    placeholder_word = st.empty()

    st.divider()
    st.markdown("""
    **How to use**
    - Hold a letter sign → stages it
    - `"next"` → adds it to the word
    - `"space"` → adds a space
    - `"del"` → deletes last letter
    """)

    if st.button("Clear Word"):
        if ctx.video_processor:
            ctx.video_processor.word = ""
            ctx.video_processor.current_letter = ""

    # ── Live polling loop ───────────────────────────────────────────────────
    # webrtc_streamer runs the camera in a background thread, so the values
    # on ctx.video_processor keep changing even though Streamlit's main
    # script only runs once per page load. This loop re-reads those values
    # every 0.3s and pushes them into the placeholders above, WITHOUT doing
    # a full st.rerun() (which would restart the camera connection).
    while ctx.state.playing:
        if ctx.video_processor:
            placeholder_letter.metric("Current Letter", ctx.video_processor.current_letter or "–")
            placeholder_word.metric("Word", ctx.video_processor.word or "–")
        time.sleep(0.3)