---
title: Fastai Petclassifier
emoji: 🐶
colorFrom: red
colorTo: red
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
---

# 🐶 Fastai Pet Classifier 🚀

A simple deep learning app built with **FastAI** and deployed using **Gradio** on **Hugging Face Spaces**. It classifies pet images (dogs and cats) into specific breeds using a fine-tuned ResNet model.

## 📸 Try It Out

👉 [Click here to launch the app](https://huggingface.co/spaces/AditiRoy171/fastai_petclassifier)

## 🧠 Model Info

- Trained on the Oxford-IIIT Pet Dataset  
- Fine-tuned using transfer learning with FastAI  
- Architecture: ResNet34  
- Exported as `pet_classifier.pkl`

## 🛠 Tech Stack

- Python  
- FastAI  
- Gradio  
- Hugging Face Spaces

## 🗂 Project Files

- `app.py` – Gradio app script  
- `pet_classifier.pkl` – Trained model file  
- `requirements.txt` – Python dependencies

## 🚀 Run Locally

To run the app on your own machine:

```bash
git clone https://github.com/AditiRoy171/pet-classifier.git
cd pet-classifier
pip install -r requirements.txt
python app.py
