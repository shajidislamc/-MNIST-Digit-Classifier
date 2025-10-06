# app.py
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# choose file you added: prefer .keras, fallback to .h5 if you want
MODEL_PATH = "mnist_model.keras"   # or "mnist_model.h5"

# load model
model = load_model(MODEL_PATH)

def preprocess_pil(img_pil):
    # img_pil is a PIL Image (RGB or RGBA) from the sketchpad
    # convert to grayscale and invert to MNIST format (white digit on black)
    img = img_pil.convert("L")                # grayscale
    img = ImageOps.invert(img)                # invert: white on black
    # crop to content bounding box to better center the digit
    bbox = img.getbbox()                      # returns None if all black
    if bbox:
        img = img.crop(bbox)
    # resize while keeping aspect ratio and pad to 28x28
    img = ImageOps.contain(img, (20, 20))     # keep digit size reasonable
    new_img = Image.new("L", (28, 28), 0)     # black background
    # paste centered
    new_img.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
    # normalize
    arr = np.array(new_img).astype("float32") / 255.0
    # check model input shape
    if model.input_shape[-1] == 784 or len(model.input_shape) == 2:
        arr = arr.reshape(1, 784)
    else:
        # assume (None, 28, 28, 1)
        arr = arr.reshape(1, 28, 28, 1)
    return arr, new_img

def predict(image):
    # image: numpy array (H, W, 3 or 4) from Gradio sketchpad
    if image is None:
        return "Draw a digit", {}
    # convert to PIL
    pil = Image.fromarray(image).convert("RGBA")
    # quick blank-check: any alpha/brightness?
    gray_check = np.array(pil.convert("L"))
    if np.max(gray_check) < 10:
        return "Draw a digit", {}
    arr, proc_img = preprocess_pil(pil)
    probs = model.predict(arr)[0]
    pred = int(np.argmax(probs))
    probs_dict = {str(i): float(probs[i]) for i in range(10)}
    # return predicted label and dictionary of probabilities
    return f"Predicted: {pred}", probs_dict

with gr.Blocks(title="MNIST Digit Classifier") as demo:
    gr.Markdown("# ✏️ MNIST Digit Classifier\nDraw a digit (0–9) below and click Predict.")
    with gr.Row():
        sketch = gr.Sketchpad(shape=(280, 280), brush_radius=12, label="Draw here")
        with gr.Column():
            btn = gr.Button("Predict")
            output_label = gr.Label(value="Draw a digit", label="Result")
            output_probs = gr.Label(value={}, label="Probabilities (0–9)")
    btn.click(fn=predict, inputs=sketch, outputs=[output_label, output_probs])

if __name__ == "__main__":
    demo.launch()
