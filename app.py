"""
Streamlit app for Customer Complaint → Product classification
Uses the exported model in ./trained_model
"""

import os
import json
import pickle
import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Optional: better GPU performance on RTX with Tensor Cores
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


@st.cache_resource
def load_model():
    """
    Load model, tokenizer, and label encoder from exported directory.
    Cached so it loads only once.
    """
    model_dir: str = "./trained_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load label encoder
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=config["num_labels"],
    )
    weights_path = os.path.join(model_dir, "model_weights.pth")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, tokenizer, label_encoder, config, device


def classify_complaint(
    text: str,
    model,
    tokenizer,
    label_encoder,
    max_length: int,
    device: torch.device,
):
    """
    Classify a single complaint and return product + confidence + top-3.
    """
    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[0]

    # Best prediction
    best_idx = int(torch.argmax(probs).item())
    best_label = label_encoder.inverse_transform([best_idx])[0]
    best_conf = float(probs[best_idx].item())

    # Top‑3 predictions
    top_k = min(3, len(label_encoder.classes_))
    top_probs, top_indices = torch.topk(probs, k=top_k)
    top_results = []
    for p, i in zip(top_probs, top_indices):
        idx = int(i.item())
        label = label_encoder.inverse_transform([idx])[0]
        top_results.append((label, float(p.item())))

    return best_label, best_conf, top_results


# ----------------- Streamlit UI ----------------- #

st.set_page_config(
    page_title="Complaint → Product Classifier",
    page_icon="📋",
    layout="centered",
)

st.title("📋 Customer Complaint Classifier")
st.markdown(
    "Enter a customer complaint and the model will classify it into a product category "
    "(e.g., credit card, mortgage, loan, etc.)."
)

# Try loading model
try:
    with st.spinner("Loading model..."):
        model, tokenizer, label_encoder, cfg, device = load_model()
    st.success(f"Model loaded on: **{device.type.upper()}**")
except Exception as e:
    st.error(
        "Could not load model from `./trained_model`. "
        "Make sure you have trained and exported it using `complaint_classifier.py`."
    )
    st.exception(e)
    st.stop()

# Input area
complaint_text = st.text_area(
    "Customer Complaints",
    height=180,
    placeholder="Example: I was charged an annual fee on my credit card without my consent...",
)


# Classify button
if st.button("🔍 Classify complaint"):
    if not complaint_text.strip():
        st.warning("Please enter a complaint first.")
    else:
        with st.spinner("Classifying..."):
            product, conf, top3 = classify_complaint(
                complaint_text,
                model,
                tokenizer,
                label_encoder,
                max_length=cfg["max_length"],
                device=device,
            )

        st.subheader("Prediction")
        st.write(f"**Department:** {product}")
        st.write(f"**Confidence:** {conf:.2%}")

        st.subheader("Top‑3 probabilities")
        for label, p in top3:
            st.write(f"- {label}: {p:.2%}")
