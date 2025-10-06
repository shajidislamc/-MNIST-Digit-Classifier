import os
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained MNIST model
model = load_model("mnist_model.h5")

def predict_digit(image):
    # If image comes as a dict from Gradio
    if isinstance(image, dict):
        image = image.get('composite')
    
    if image is None:
        return "Please draw a digit"
    
    # Convert to grayscale PIL image
    img = Image.fromarray(image).convert("L")
    # Invert colors for MNIST style
    img = Image.eval(img, lambda x: 255 - x)
    img = img.resize((28, 28))
    
    img_array = np.array(img).reshape(1, 784) / 255.0
    
    # Predict
    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction))
    
    # Optionally, return probabilities too
    probabilities = {str(i): float(prediction[0][i]) for i in range(10)}
    
    return f"Predicted Digit: {predicted_digit}\nProbabilities: {probabilities}"

# Gradio Interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="numpy"),
    outputs="text",
    title="MNIST Digit Classifier",
    description="Draw a digit (0-9) and see the model's prediction."
)

# Use Render's dynamic port
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 5000))
)
