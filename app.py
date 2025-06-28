from fastai.vision.all import *
import gradio as gr
import os

# Load the model
learn = load_learner('pet_classifier.pkl')

# Define prediction function
def classify_pet(img):
    pred, idx, probs = learn.predict(img)
    return f"Prediction: {pred} (Confidence: {probs[idx]:.2f})"

# Automatically load images from 'img/' directory
image_folder = "img"
examples = [[os.path.join(image_folder, fname)] for fname in os.listdir(image_folder)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]

description = """
**Example Images:**  
Click on any image below to try it!
"""

# Create Gradio interface
demo = gr.Interface(
    fn=classify_pet,
    inputs=gr.Image(type="pil"),
    outputs="text",
    examples=examples,
    title="FastAI Pet Classifier",
    description=description
)

# Launch app
demo.launch()
