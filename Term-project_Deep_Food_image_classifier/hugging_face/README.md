---
license: mit
tags:
  - image-classification
  - food-classification
  - tensorflow
  - keras
  - deep-learning
  - computer-vision
  - resnet
  - evolutionary-algorithm
datasets:
  - food101
metrics:
  - accuracy
pipeline_tag: image-classification
library_name: keras
---

# 🍔 American Food Image Classifier

A 10-class American food image classifier trained with **ES(1+1) evolutionary hyperparameter optimization**, achieving **79.80% test accuracy**.

## Model Description

This model classifies images of American food into 10 categories using a custom ResNet-style CNN architecture with residual connections. The hyperparameters were optimized using the ES(1+1) evolutionary strategy.

### Architecture Details

- **Type:** Custom ResNet-style CNN with 4 residual blocks
- **Input Size:** 224×224 RGB images
- **Output:** 10-class softmax probabilities
- **Parameters:** ~20.6M
- **Framework:** TensorFlow 2.x / Keras

### Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001381 (ES-optimized) |
| Dropout Rate | 0.256 (ES-optimized) |
| Label Smoothing | 0.30 (ES-optimized) |
| Epochs | 50 |
| Batch Size | 24 |
| Data Augmentation | Random flip, rotation, zoom, contrast |

## Supported Classes

The model can classify the following 10 American food types:

| Index | Class Name |
|-------|------------|
| 0 | chicken_wings 🍗 |
| 1 | churros 🥖 |
| 2 | french_fries 🍟 |
| 3 | hamburger 🍔 |
| 4 | hot_dog 🌭 |
| 5 | ice_cream 🍦 |
| 6 | macaroni_and_cheese 🧀 |
| 7 | pancakes 🥞 |
| 8 | pizza 🍕 |
| 9 | waffles 🧇 |

## Usage

### Quick Start

```python
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="HAR5HA-YELLELA/american_food_classifier",
    filename="optimized_model_best.keras"
)

# Load model
model = tf.keras.models.load_model(model_path)

# Class names
classes = [
    "chicken_wings", "churros", "french_fries", "hamburger", "hot_dog",
    "ice_cream", "macaroni_and_cheese", "pancakes", "pizza", "waffles"
]

# Load and preprocess image
img = Image.open("your_food_image.jpg").resize((224, 224))
img_array = np.array(img) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_batch)[0]
predicted_class = classes[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

### Inference from URL

```python
import requests
from io import BytesIO

def predict_from_url(url, model, classes, threshold=0.6):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_batch, verbose=0)[0]
    top_idx = np.argmax(predictions)
    confidence = predictions[top_idx]
    
    if confidence >= threshold:
        return classes[top_idx], confidence
    else:
        return "out_of_scope", confidence

# Example
result, conf = predict_from_url("https://example.com/pizza.jpg", model, classes)
print(f"{result}: {conf:.2%}")
```

## Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 79.80% |
| **Validation Accuracy** | 79.15% |
| **Improvement over Baseline** | +8.13% |

### Comparison with Other Methods

| Method | Test Accuracy | Improvement |
|--------|---------------|-------------|
| Baseline CNN | 73.80% | - |
| Keras Tuner (Hyperband) | 79.27% | +7.41% |
| **ES(1+1) (This Model)** | **79.80%** | **+8.13%** |

## Out-of-Scope Detection

The model includes confidence-based out-of-scope detection. Images with prediction confidence below 60% are flagged as potentially out-of-scope (not one of the 10 trained food categories).

```python
threshold = 0.6
if confidence < threshold:
    print("⚠️ Image may be out of scope")
```

## Limitations

- Only trained on 10 American food categories
- May not generalize well to non-American cuisines
- Performance may degrade on heavily processed or stylized images
- Best results with centered, well-lit food photographs

## Training Data

Trained on a subset of the Food-101 dataset:
- **Total Images:** 10,000 (1,000 per class)
- **Train/Val/Test Split:** 70% / 15% / 15%
- **Image Size:** 224×224 pixels

## Citation

If you use this model, please cite:

```bibtex
@misc{yellela2025foodclassifier,
  author = {Yellela, V. Harsha Vardhan},
  title = {American Food Image Classifier with ES(1+1) Optimization},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/HAR5HA-YELLELA/american_food_classifier}
}
```

## Author

**V. Harsha Vardhan Yellela**  
Lawrence Technological University  
MCS-5993: Evolutionary Computing & Deep Learning  
Fall 2025
