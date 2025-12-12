"""
Gradio App for American Food Classifier
Deploy this on Hugging Face Spaces for a web API

This file is for OPTIONAL Spaces deployment (not required for model upload)
"""

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from huggingface_hub import hf_hub_download

# =============================================================================
# CONFIGURATION
# =============================================================================
REPO_ID = "HAR5HA-YELLELA/american_food_classifier"
MODEL_FILE = "optimized_model_best.keras"

# Class names
CLASSES = [
    "chicken_wings", "churros", "french_fries", "hamburger", "hot_dog",
    "ice_cream", "macaroni_and_cheese", "pancakes", "pizza", "waffles"
]

CLASS_EMOJIS = {
    "chicken_wings": "🍗",
    "churros": "🥖", 
    "french_fries": "🍟",
    "hamburger": "🍔",
    "hot_dog": "🌭",
    "ice_cream": "🍦",
    "macaroni_and_cheese": "🧀",
    "pancakes": "🥞",
    "pizza": "🍕",
    "waffles": "🧇"
}

# =============================================================================
# LOAD MODEL
# =============================================================================
print("🔄 Loading model from Hugging Face Hub...")

try:
    # Download model from Hub
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Trying local file...")
    model = tf.keras.models.load_model(MODEL_FILE)

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_food(image):
    """
    Predict food class from uploaded image
    
    Args:
        image: numpy array from Gradio Image component
        
    Returns:
        Dictionary of class probabilities for Gradio Label component
    """
    if image is None:
        return {"No image provided": 1.0}
    
    # Preprocess image
    img = Image.fromarray(image).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_batch, verbose=0)[0]
    
    # Create result dictionary with emojis
    result = {}
    for i, cls in enumerate(CLASSES):
        emoji = CLASS_EMOJIS.get(cls, "")
        label = f"{emoji} {cls.replace('_', ' ').title()}"
        result[label] = float(predictions[i])
    
    return result

# =============================================================================
# GRADIO INTERFACE
# =============================================================================
demo = gr.Interface(
    fn=predict_food,
    inputs=gr.Image(label="Upload Food Image", type="numpy"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="🍔 American Food Classifier",
    description="""
    Upload an image of American food to classify it into one of 10 categories.
    
    **Supported Foods:** Chicken Wings, Churros, French Fries, Hamburger, Hot Dog, 
    Ice Cream, Mac & Cheese, Pancakes, Pizza, Waffles
    
    **Model:** Custom ResNet-style CNN trained with ES(1+1) evolutionary optimization
    **Accuracy:** 79.80% on test set
    """,
    article="""
    ## About This Model
    
    This classifier was trained on a subset of the Food-101 dataset using a custom 
    ResNet-style CNN architecture. Hyperparameters were optimized using the ES(1+1) 
    evolutionary strategy, achieving 79.80% test accuracy.
    
    ### Author
    V. Harsha Vardhan Yellela | Lawrence Technological University | MCS-5993
    
    ### API Usage
    ```python
    from gradio_client import Client
    
    client = Client("HAR5HA-YELLELA/american_food_classifier")
    result = client.predict(image="food.jpg", api_name="/predict")
    print(result)
    ```
    """,
    examples=[
        # Add example images if available
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# =============================================================================
# LAUNCH
# =============================================================================
if __name__ == "__main__":
    demo.launch()
