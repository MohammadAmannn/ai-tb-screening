# test_model.py
import os
import sys

print("🔍 Checking project structure...")
print("Current directory:", os.getcwd())
print("\nFiles in current directory:")
for item in os.listdir('.'):
    print(f"  - {item}")

print("\n📁 Model folder contents:")
if os.path.exists('model'):
    for item in os.listdir('model'):
        print(f"  - {item}")
else:
    print("  Model folder not found!")

print("\n🧪 Testing model loading...")
try:
    from predict import TBClassifier
    
    # Try to load model
    detector = TBClassifier()
    print("✅ Model loaded successfully!")
    
    # Check model summary
    print(f"\n📋 Model input shape: {detector.model.input_shape}")
    print(f"📋 Model output shape: {detector.model.output_shape}")
    
    # Test with dummy image
    import numpy as np
    dummy_image = np.random.rand(224, 224, 3)
    cv2.imwrite('test_dummy.jpg', dummy_image * 255)
    
    result = detector.predict('test_dummy.jpg')
    print(f"\n🧪 Dummy test prediction: {result}")
    
    # Clean up
    if os.path.exists('test_dummy.jpg'):
        os.remove('test_dummy.jpg')
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Setup check complete!")