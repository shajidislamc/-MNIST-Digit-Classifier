import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained MNIST model
model = load_model("mnist_model.h5")

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("✏️ MNIST Digit Classifier")
st.write("Draw a digit (0-9) below and let the model predict it!")

# --- Drawing input ---
st.subheader("Draw a digit (28x28 pixels)")

# Streamlit's file uploader for drawing
uploaded_file = st.file_uploader("Upload a handwritten digit image (PNG/JPG)", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("L")  # convert to grayscale
    img = ImageOps.invert(img)  # invert colors for MNIST style
    img = img.resize((28, 28))
    
    st.image(img, caption="Your Input Digit", width=150)
    
    # Preprocess for model
    img_array = np.array(img).reshape(1, 28*28) / 255.0
    
    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    st.success(f"🖊️ The model predicts this digit as: **{predicted_digit}**")
