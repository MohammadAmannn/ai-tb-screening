# test_model_performance.py
import tensorflow as tf
import numpy as np
import cv2
import os

print("🧪 TESTING MODEL PERFORMANCE")
print("="*50)

# Load model
model = tf.keras.models.load_model("model/tb_model.keras")

# Test 1: Extreme cases
print("\n1️⃣ Testing extreme cases:")
test_cases = [
    ("Black image", np.zeros((1, 192, 192, 3))),
    ("White image", np.ones((1, 192, 192, 3))),
    ("Gray image", np.full((1, 192, 192, 3), 0.5)),
    ("Random image", np.random.rand(1, 192, 192, 3))
]

for name, img in test_cases:
    pred = model.predict(img, verbose=0)[0][0]
    print(f"   {name}: {pred:.4f} → {'TB' if pred > 0.5 else 'Normal'}")

# Test 2: Check if model is broken
print("\n2️⃣ Checking model behavior:")
predictions = []
for i in range(10):
    test_img = np.random.rand(1, 192, 192, 3)
    pred = model.predict(test_img, verbose=0)[0][0]
    predictions.append(pred)

avg_pred = np.mean(predictions)
print(f"   Average prediction on random images: {avg_pred:.4f}")

if 0.48 <= avg_pred <= 0.52:
    print("   ⚠️ WARNING: Model is predicting ~0.5 for everything!")
    print("   This means it's NOT learning to distinguish TB from Normal")
else:
    print("   ✅ Model shows variation in predictions")

print("\n" + "="*50)