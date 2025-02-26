
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

# Load YOLOv8 model
model = YOLO('weights/yolov8n.pt')

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #f0f0f5;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #2E86C1;
        animation: fadeIn 2s;
        margin-top: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #e6e6e6;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #d6d6d6;
        animation: slideIn 1s;
    }
    .sidebar .sidebar-content h2 {
        color: #2E86C1;
    }
    .menu-item {
        font-size: 18px;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
        background-color: #FF6347;
        color: #fff;
        margin: 10px 0;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .menu-item:hover {
        background-color: #E5533C;
    }
    .detected-objects {
        background-color: #fff;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin-top: 20px;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateX(-200px); }
        to { transform: translateX(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app interface
st.markdown('<div class="title">Object Detection System</div>', unsafe_allow_html=True)

# Sidebar menu
menu_items = ["Home", "How to Use", "About the Project", "Developed By"]
selected_menu = st.sidebar.selectbox("Menu", menu_items)

# Page content based on menu selection
if selected_menu == "Home":
    st.subheader("Welcome to the Object Detection System")
    st.write("""
        - Introduction to the Object Detection System

    """)

    # Select input source
    source = st.selectbox("Select Input Source", ["Image", "Video", "Webcam"])

    # Set model confidence
    confidence = st.slider("Model Confidence", 0.1, 1.0, 0.5)

    # List to store detected objects
    detected_objects = []

    # Function to process the results and get detected objects
    def process_results(results, img_array):
        objects = []
        for result in results:
            for bbox in result.boxes:
                x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                label = model.names[int(bbox.cls)]
                conf = float(bbox.conf)  # Convert tensor to float
                objects.append((label, conf, (x1, y1, x2, y2)))
                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_array, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return objects

    # Process input source
    if source == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            img_array = np.array(image)
            results = model(img_array, conf=confidence)
            detected_objects = process_results(results, img_array)
            st.image(img_array, caption='Detected Objects', use_column_width=True)

    elif source == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            ret, frame = cap.read()
            if ret:
                results = model(frame, conf=confidence)
                detected_objects = process_results(results, frame)
                st.image(frame, caption='Detected Objects', use_column_width=True)
            cap.release()

    elif source == "Webcam":
        run = st.checkbox('Run Webcam')
        if run:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret:
                results = model(frame, conf=confidence)
                detected_objects = process_results(results, frame)
                st.image(frame, caption='Detected Objects', use_column_width=True)
            cap.release()

    # Display the list of detected objects
    if detected_objects:
        st.subheader("Detected Objects:")
        st.markdown('<div class="detected-objects">', unsafe_allow_html=True)
        for obj in detected_objects:
            label, conf, bbox = obj
            st.write(f"Object: {label}, Confidence: {conf:.2f}, Bounding Box: {bbox}")
        st.markdown('</div>', unsafe_allow_html=True)

elif selected_menu == "How to Use":
    st.subheader("How to Use the Object Detection System")
    st.write("""
        1. **Click on given link to acess on Browser** (sensors, cameras, control unit)
        2. **Select the Input Resource** to your home network
        3. **Set model Confidence level** via the mobile app or web interface
        4. **Select image, Video, Webcam** in real-time through the app
        5. **Receive alerts and notifications** for any detected intrusions
    """)

elif selected_menu == "About the Project":
    st.subheader("About the Object Detection System Project")
    st.write("""
        - **Objective:** To Easily Detect Every Object in Image or Video.
        - **Components Used:** Computer, Laptop, Smart Phone.
        - **Technology:** YOLO V8, Streamlit, OpenCV, PIL More
        - **Features:** Real-time monitoring, intrusion detection, Accuracy, remote access
        - **Benefits:** Enhanced security, peace of mind, easy installation and use
    """)

elif selected_menu == "Developed By":
    st.subheader("Meet the Developers")
    st.write("""
        - **Paras Longadge:** Project Leader
        - **Pranay Dhore:** Lead Developer
        - **Sanket Tajne** UI/UX Designer
        - **Mohit Barse:** PPT Creation and Present
        - **Kshitij Deshmukh:** Software Work
        
    """)


