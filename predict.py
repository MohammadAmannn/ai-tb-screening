# src/predict.py — FINAL, CONSISTENT WITH TRAINING

import tensorflow as tf
import numpy as np
import cv2
import os

# ================= SAME FUNCTION AS TRAINING =================
@tf.keras.utils.register_keras_serializable()
def force_grayscale_fn(image):
    gray = tf.image.rgb_to_grayscale(image)
    return tf.image.grayscale_to_rgb(gray)

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tb_detection_model.keras")
IMG_SIZE = (224, 224)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"force_grayscale_fn": force_grayscale_fn},
    compile=False
)

# ================= PREPROCESS (MATCH TRAINING EXACTLY) =================
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image file")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    img = tf.convert_to_tensor(img, dtype=tf.float32)

    # 🔴 MUST MATCH TRAINING
    img = force_grayscale_fn(img)
    img = tf.keras.applications.resnet.preprocess_input(img)

    img = tf.expand_dims(img, axis=0)
    return img

# ================= PREDICTION =================
def predict_tb(image_path):
    img = preprocess_image(image_path)

    prob = float(model.predict(img, verbose=0)[0][0])
    confidence = max(prob, 1 - prob)

    if prob >= 0.70:
        risk = "High TB Risk"
        advice = "Urgent confirmatory testing recommended."
    elif prob >= 0.45:
        risk = "Moderate TB Risk"
        advice = "Clinical evaluation advised."
    else:
        risk = "Low TB Risk"
        advice = "No immediate TB indicators detected."

    return {
        "probability": round(prob, 3),
        "confidence": round(confidence, 3),
        "risk": risk,
        "advice": advice,
        "predicted_class": "Tuberculosis" if prob >= 0.5 else "Normal"
    }
