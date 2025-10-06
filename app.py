import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model("mnist_model.h5")

def predict_digit(image):
    if isinstance(image, dict):
        image = image.get('composite')
    
    if image is None:
        return "Please draw a digit"
    
    img = Image.fromarray(image).convert("L")
    img = Image.eval(img, lambda x: 255 - x)
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, 784) / 255.0
    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction))
    
    # Return only the predicted digit
    return f"Predicted Digit: {predicted_digit}"

# Gradio interface
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(type="numpy"),
    outputs="text",   # Make sure output is text
    title="MNIST Digit Classifier",
    description="Draw a digit (0-9) and see the model's prediction."
)

demo.launch(server_name="0.0.0.0", server_port=5000)
