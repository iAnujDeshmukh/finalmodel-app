import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.xception import preprocess_input
from mtcnn import MTCNN
import time
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parameters
TIME_STEPS = 30  # Frames per video
HEIGHT, WIDTH = 299, 299

# Model builder
def build_model(lstm_hidden_size=256, num_classes=2, dropout_rate=0.5):
    inputs = layers.Input(shape=(TIME_STEPS, HEIGHT, WIDTH, 3))
    base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')
    x = layers.TimeDistributed(base_model)(inputs)
    x = layers.LSTM(lstm_hidden_size)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

# Load model
model_path = 'COMBINED_best_Phase1.keras'
model = build_model()
model.load_weights(model_path)

# Extract faces
def extract_faces_from_video(video_path, num_frames=TIME_STEPS):
    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    idx = 0
    success = True
    faces_found = False  # Track if faces are found

    while success and len(frames) < num_frames:
        success, frame = cap.read()
        if not success:
            break
        if idx in frame_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(frame_rgb)
            if detections:
                faces_found = True  # Mark that at least one face is found
                x, y, width, height = detections[0]['box']
                x, y = max(0, x), max(0, y)
                x2, y2 = x + width, y + height
                face = frame_rgb[y:y2, x:x2]
                face_image = Image.fromarray(face).resize((WIDTH, HEIGHT))
                face_array = preprocess_input(np.array(face_image))
                frames.append(face_array)
            else:
                frames.append(np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32))
        idx += 1

    cap.release()

    # If no faces were found, raise a flag
    if not faces_found:
        return None, None  # Return None to indicate no faces detected

    video_array = np.expand_dims(np.array(frames), axis=0)
    return video_array, frames

# Prediction
def make_prediction(video_file):
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())
    video_array, frames = extract_faces_from_video("temp_video.mp4")
    if video_array is None:  # Check if no faces were detected
        return None, None, None
    predictions = model.predict(video_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    probabilities = predictions[0]
    return predicted_class, probabilities, frames

# Streamlit UI
st.set_page_config(page_title="VARS - Video Analysis", layout="wide")
st.markdown("<style>h1{font-size: 45px !important;}</style>", unsafe_allow_html=True)

# Header Section
image = Image.open("Image2.png")
desired_height = 600
aspect_ratio = image.width / image.height
new_width = int(desired_height * aspect_ratio)
resized_image = image.resize((new_width, desired_height))
st.image(resized_image, use_container_width=True)

st.title("🎥 VARS: Video Analysis for Realness & Synthetic Detection")
st.markdown(
    """
Welcome to **VARS**, your DeepFake detection companion!  
Upload a video, and we'll analyze it using state-of-the-art deep learning models to determine if it's real or fake.
""",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Steps")
st.sidebar.markdown(
    """
1. **Upload Video:** Choose a video file (mp4, mov, avi).  
2. **Process Frames:** Detect faces and preprocess.  
3. **Analyze:** Our AI model predicts 'Real' or 'Fake'.  
4. **View Results:** Get detailed predictions.
"""
)
st.sidebar.info("Made by Group 1: Sarvansh, Vansh, Ranveer, and Anuj ✨")

# Upload video
video_file = st.file_uploader("🎥 Upload Your Video File", type=["mp4", "mov", "avi"])

if video_file is not None:
    st.video(video_file)
    start_time = time.time()

    # Loading animation
    with st.spinner("🚀 Processing video... Please wait!"):
        predicted_class, probabilities, frames = make_prediction(video_file)

    if predicted_class is None:  # No faces detected
        st.error("No faces detected in the uploaded video. Please upload a different video.")
    else:
        end_time = time.time()
        processing_time = end_time - start_time

        # Display results
        st.subheader("🎯 Results")
        if predicted_class == 0:
            st.success("The video is classified as **Real**!")
        else:
            st.error("The video is classified as **Fake**!")

        st.write(f"**Prediction Confidence:**")
        st.progress(int(probabilities[predicted_class] * 100))

        # Tabs for detailed results
        tab1, tab2, tab3 = st.tabs(["📊 Probabilities", "🖼️ Frame Previews", "⏱️ Processing Time"])

        with tab1:
            st.subheader("Class Probabilities")
            st.bar_chart({"Real": [probabilities[0]], "Fake": [probabilities[1]]})

        with tab2:
            st.subheader("Frame Previews")
            st.write("Key frames analyzed during the process:")
            cols = st.columns(5)
            for i, frame in enumerate(frames[:10]):
                frame = np.clip(frame, 0, 1)
                frame = (frame * 255).astype(np.uint8)
                with cols[i % 5]:
                    st.image(frame, caption=f"Frame {i+1}", use_container_width=True)

        with tab3:
            st.subheader("Processing Details")
            st.write(f"**Time Taken:** {processing_time:.2f} seconds")
