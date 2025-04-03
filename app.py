import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile

# Load YOLOv8 model
@st.cache_resource
def load_model():
    model_path = 'weights/yolov8n.pt'  # Specify the path to the YOLOv8 weights file
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_model()

# Streamlit app interface
st.title("Object Detection System")

# Sidebar menu
menu_items = ["Home", "Live Detection", "About", "Developers"]
selected_menu = st.sidebar.selectbox("Menu", menu_items)

# Process detection results
def process_results(results, image):
    detected_objects = []
    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            label = model.names[int(bbox.cls)]
            conf = float(bbox.conf)
            detected_objects.append((label, conf, (x1, y1, x2, y2)))
            # Draw bounding box and label on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return detected_objects

# Home menu
if selected_menu == "Home":
    st.subheader("Welcome to the Object Detection System")
    source = st.selectbox("Choose an input source:", ["Image", "Video"])
    confidence = st.slider("Model Confidence Threshold", 0.1, 1.0, 0.5)

    if source == "Image":
        uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
        if uploaded_image and model:
            image = Image.open(uploaded_image)
            img_array = np.array(image)
            results = model(img_array, conf=confidence)
            detected_objects = process_results(results, img_array)
            st.image(img_array, caption="Detected Objects", use_container_width=True)
            st.write("Detected Objects:")
            for obj in detected_objects:
                st.write(f"Object: {obj[0]}, Confidence: {obj[1]:.2f}, Box: {obj[2]}")

    elif source == "Video":
        uploaded_video = st.file_uploader("Upload a video:", type=["mp4", "avi"])
        if uploaded_video and model:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_video.read())
                cap = cv2.VideoCapture(temp_file.name)
                stframe = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame, conf=confidence)
                    process_results(results, frame)
                    stframe.image(frame, channels="BGR", use_container_width=True)
                cap.release()

# Live Detection menu
elif selected_menu == "Live Detection":
    st.subheader("Live Detection using Webcam")
    run_webcam = st.checkbox("Run Webcam")
    if run_webcam and model:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image.")
                break
            results = model(frame)
            process_results(results, frame)
            stframe.image(frame, channels="BGR", use_container_width=True)
        cap.release()

# About menu
elif selected_menu == "About":
    st.subheader("About This Project")
    st.write("""
    This project demonstrates an Object Detection System built using YOLOv8 and Streamlit.
    You can use it to detect objects in images, videos, or live webcam streams.
    """)

# Developers menu
elif selected_menu == "Developers":
    st.subheader("Developed By")
    st.write("""
    - **Dr. Suhashini Chaurasiya** (Guide)
    - **Paras Longadge**: Project Leader
    - **Pranay Dhore**: Lead Developer
    - **Sanket Tajne**: UI/UX Designer
    - **Mohit Barse**: Presentation Creator
    - **Kshitij Deshmukh**: Software Developer
    """)

# Instructions for errors
if not model:
    st.error("Model not loaded. Check the weights path or deployment environment.")
