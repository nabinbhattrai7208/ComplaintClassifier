# Customer Complaint Classifier - Trained Model

## Contents

- `model_weights.pth`: PyTorch model state dictionary
- `tokenizer/`: HuggingFace tokenizer files
- `label_encoder.pkl`: Scikit-learn label encoder (product mappings)
- `model_config.json`: Model configuration and metadata

## Usage

```python
from inference import load_model, predict

# Load model
model, tokenizer, label_encoder = load_model("./trained_model")

# Predict
complaint = "My credit card was charged without authorization"
result = predict(model, tokenizer, label_encoder, complaint)
print(f"Product: {result['product']}, Confidence: {result['confidence']:.2%}")
```

## Model Details

- Architecture: RoBERTa-base
- Training Data: 2.3M customer complaints
- Product Categories: 21 classes
- Expected Accuracy: 88-93%
- Max Sequence Length: 256 tokens

## File Sizes

- Model weights: ~440MB
- Tokenizer: ~1MB
- Label encoder: <1MB
- Total: ~441MB
