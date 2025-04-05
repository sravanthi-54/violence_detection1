import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("modelnew.h5")

# Function to detect violence in a frame
def detect_violence(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (128, 128))  # Resize for model
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    return prediction[0][0]  # Confidence score

# Function to process video and extract most violent frame
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    best_frame = None
    max_confidence = 0
    violence_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        confidence = detect_violence(frame)

        if confidence > 0.5:  # If violence is detected
            violence_detected = True
            if confidence > max_confidence:  # Store frame with highest confidence
                max_confidence = confidence
                best_frame = frame

    cap.release()

    if violence_detected and best_frame is not None:
        return Image.fromarray(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)), max_confidence
    else:
        return None, None

# Streamlit UI
st.set_page_config(layout="wide")  
st.title("Violence Detection System")

# Sidebar for choosing mode
option = st.sidebar.radio("Choose Mode:", [" Live Camera", " Upload Video"])

if option == " Live Camera":
    st.subheader(" Live Violence Detection")

    # Buttons for live video
    start_webcam = st.button(" Start Live Video")
    stop_webcam = st.button(" Stop Live Video")

    # Video display area
    stframe = st.empty()

    if start_webcam:
        cap = cv2.VideoCapture(0)  # Open webcam

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error(" Could not access the webcam.")
                break

            # Resize frame to a smaller size
            frame = cv2.resize(frame, (640, 480))

            # Violence detection
            confidence = detect_violence(frame)
            violence_text = "Violence: True" if confidence > 0.5 else "Violence: False"
            color = (0, 0, 255) if confidence > 0.5 else (0, 255, 0)  # Red for True, Green for False

            # Overlay text on frame
            cv2.putText(frame, f"{violence_text} ({confidence*100:.2f}%)", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Convert frame to RGB and update Streamlit UI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", width=640)  # Fixed width

            # Stop when button is pressed
            if stop_webcam:
                break

        cap.release()
        cv2.destroyAllWindows()
        st.success(" Live video stopped.")

elif option == " Upload Video":
    st.subheader(" Upload Video for Violence Detection")

    # Upload video
    video_file = st.file_uploader(" Upload a Video", type=["mp4", "mov", "avi"])

    if video_file is not None:
        # Save uploaded video
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video_file.read())

        # Split page into two equal columns
        col1, col2 = st.columns([1, 1])

        # Left Column: Input Video
        with col1:
            st.subheader(" Input Video")
            st.video("uploaded_video.mp4")

        # Right Column: Output Frame
        with col2:
            st.subheader(" Violence Detection Output")

            # Process video
            best_frame_image, max_confidence = process_video("uploaded_video.mp4")

            if best_frame_image is not None:
                st.image(best_frame_image, caption=f" Detected Frame (Confidence: {max_confidence*100:.2f}%)", use_column_width=True)
                st.error(f" Violence Detected! Confidence: {max_confidence*100:.2f}%")
                
                # Display progress bar for violence confidence
                st.progress(float(max_confidence))  # Display the confidence as a progress bar
                st.text(f"Confidence: {max_confidence*100:.2f}%")
                
                # Display message for violent content
                st.warning(" Violent Content! Avoid for kids. Not safe to use.")
            else:
                st.success(" No violence detected in the video.")
                
                # Display message for safe content
                st.info(" Kids Friendly! Safe to use on all platforms.")
