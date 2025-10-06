import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load the trained MNIST model (.keras format recommended)
model = load_model("mnist_model.keras")

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("✏️ MNIST Digit Classifier")
st.write("Draw a digit (0-9) below and get the prediction!")

# --- Canvas for drawing ---
canvas_result = st_canvas(
    fill_color="#000000",      # Black brush
    stroke_width=15,
    stroke_color="#FFFFFF",    # White digit (like MNIST)
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    # Check if user drew anything using the alpha channel
    alpha_channel = canvas_result.image_data[:, :, 3]
    if np.max(alpha_channel) > 0:  # User drew something
        # Convert to grayscale PIL image
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
        img = Image.eval(img, lambda x: 255 - x)  # Invert colors
        img = img.resize((28, 28))
        
        # Show processed image
        st.image(img, caption="Processed Input (28x28)", width=150)
        
        # Preprocess for model
        img_flat = np.array(img).reshape(1, 784) / 255.0
        
        # Predict
        prediction = model.predict(img_flat)
        predicted_digit = np.argmax(prediction)
        
        st.success(f"🖊️ The model predicts this digit as: **{predicted_digit}**")
    else:
        st.info("✏️ Draw a digit above to get a prediction!")
