import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load your trained model
model = load_model("mnist_model.h5")

st.title("MNIST Digit Classifier 🎨")
st.write("Draw a digit (0–9) below and click Predict to see the model's guess.")

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas"
)

# Predict button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas image to grayscale and resize
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8')).resize((28, 28))
        img = np.array(img) / 255.0
        img = img.reshape(1, 784)
        prediction = model.predict(img).argmax()
        st.success(f"Predicted Digit: {prediction}")
    else:
        st.warning("Please draw something before predicting.")
