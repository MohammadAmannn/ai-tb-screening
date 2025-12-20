import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

MODEL_PATH = "model/tb_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img / 255.0

    img = np.expand_dims(img, axis=0)
    return img

def predict_tb(image_path):
    img = preprocess_image(image_path)

    prob = model.predict(img)[0][0]

    # Screening thresholds (IMPORTANT)
    if prob >= 0.7:
        risk = "High TB Risk"
    elif prob >= 0.4:
        risk = "Moderate TB Risk"
    else:
        risk = "Low TB Risk"

    return risk, float(prob)
