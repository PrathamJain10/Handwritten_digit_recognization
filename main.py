import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# -------------------------------
# Load Model (Using st.cache_resource)
# -------------------------------
@st.cache_resource
def load_mnist_model():
    model = load_model('best_mnist_model.h5')  
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model = load_mnist_model()

# -------------------------------
# App Title and Description
# -------------------------------
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (JPEG/PNG) and let the model predict the digit.")

# -------------------------------
# File Uploader
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# -------------------------------
# Preprocess Image (Using st.cache_data) 
# -------------------------------
@st.cache_data
def preprocess_image(image_array, invert=False):
    """Resize to 28x28, normalize, and reshape for prediction."""
    img_array = image_array.astype('float32')
    if invert:
        img_array = 255 - img_array  # Invert colors if needed
    
    img_array /= 255.0  # Normalize to range [0,1]
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)  # ✅ FIXED

    # Convert image to grayscale and resize
    image_gray = ImageOps.grayscale(image)
    image_resized = image_gray.resize((28, 28))
    
    # Convert PIL image to NumPy array
    img_array = np.array(image_resized)
    
    # Option to invert colors
    invert = st.checkbox("Invert image colors (if digit is dark on light background)")
    
    # Preprocess image
    processed_image = preprocess_image(img_array, invert)

    # -------------------------------
    # Predict on Button Click
    # -------------------------------
    if st.button("Predict Digit"):
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction, axis=1)[0]
        st.success(f"Predicted Digit: {predicted_digit}")
