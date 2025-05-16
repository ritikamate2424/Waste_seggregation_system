import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from PIL import Image
import base64

# Function to Set Background Image
def set_bg(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load Background Image
set_bg("iws.png")  # Ensure "iws.png" is in the same folder

# Load the Trained CNN Model
@st.cache_resource
def load_model():
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.6),
        Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('best_waste_segregation_model.h5')
    
    return model

model = load_model()

# Define Waste Class Labels
class_labels = ["Organic", "Recyclable"]

# Function for Image Preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

# UI Styling
st.markdown(
    """
    <style>
    .file-uploader {
        border: 2px dashed #ffffff;
        padding: 20px;
        text-align: center;
        font-size: 18px;
        font-family: 'Times New Roman', serif;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }

    .center-text {
        text-align: center;
        font-family: 'Times New Roman', serif;
        color: white;
    }

    .prediction-box {
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: white;
        background: linear-gradient(to right, #00ff00, #00aaff);
        padding: 10px;
        border-radius: 10px;
        display: inline-block;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h1 class='center-text'>Waste Segregation System</h1>", unsafe_allow_html=True)

# File Uploader
st.markdown("<div class='file-uploader'>Drag and drop an image here or click to upload</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.markdown("<h3 class='center-text'>Uploaded Image:</h3>", unsafe_allow_html=True)
    st.image(image, caption="", use_container_width=True)

    # Preprocess Image & Make Prediction
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)

    # Display Results
    st.markdown(f"<div class='prediction-box'>Waste Classification: {class_labels[predicted_class]}</div>", unsafe_allow_html=True)

# Camera Input for Capturing Images
camera_image = st.camera_input("Capture an image with your webcam")

if camera_image is not None:
    img_array = np.frombuffer(camera_image.getvalue(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    st.markdown("<h3 class='center-text'>Captured Image:</h3>", unsafe_allow_html=True)
    st.image(image_pil, caption="", use_container_width=True)

    # Preprocess & Predict
    preprocessed_img = preprocess_image(image_pil)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)

    # Display Results
    st.markdown(f"<div class='prediction-box'>Waste Classification: {class_labels[predicted_class]}</div>", unsafe_allow_html=True)
