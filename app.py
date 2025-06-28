from fastai.vision.all import *
import gradio as gr

# Load the model (your model file is pet_classifier.pkl)
learn = load_learner('pet_classifier.pkl')

# Define prediction function
def classify_pet(img):
    pred, idx, probs = learn.predict(img)
    return f"Prediction: {pred} (Confidence: {probs[idx]:.2f})"

# Create Gradio interface
demo = gr.Interface(
    fn=classify_pet,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="FastAI Pet Classifier",
    description="Upload a cat or dog image and get the predicted breed."
)

# Launch app
demo.launch()
