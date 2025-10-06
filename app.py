import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load the trained MNIST model (saved in .keras format)
model = load_model("mnist_model.keras")

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
    # Convert to grayscale image
    img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA').convert('L')
    # Invert colors
    img = Image.eval(img, lambda x: 255 - x)
    
    # Only proceed if user drew something (check pixel intensity)
    img_array_check = np.array(img)
    if np.max(img_array_check) > 10:  # threshold to detect any drawing
        # Resize for model
        img = img.resize((28, 28))
        st.image(img, caption="Processed Input (28x28)", width=150)
        
        # Preprocess for model
        img_array = np.array(img).reshape(1, 784) / 255.0
        
        # Predict
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        
        st.success(f"🖊️ The model predicts this digit as: **{predicted_digit}**")
    else:
        st.info("✏️ Draw a digit above to get a prediction!")
