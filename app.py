import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import cv2

# Load the trained MNIST model
model = load_model("mnist_model.h5")

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("✏️ MNIST Digit Classifier")
st.write("Draw a digit (0-9) below and see the model prediction in real-time!")

# --- Canvas for drawing ---
canvas_result = st_canvas(
    fill_color="#000000",  # Black brush
    stroke_width=15,
    stroke_color="#FFFFFF",  # White digit (like MNIST)
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    # Convert to grayscale 28x28
    img = cv2.cvtColor(canvas_result.image_data.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0  # normalize
    img_flat = img.reshape(1, 784)

    # Predict
    prediction = model.predict(img_flat)
    predicted_digit = np.argmax(prediction)

    st.success(f"🖊️ The model predicts this digit as: **{predicted_digit}**")

    # Show prediction probabilities as a bar chart
    st.subheader("Prediction Probabilities")
    prob_df = {str(i): float(prediction[0][i]) for i in range(10)}
    st.bar_chart(prob_df)
