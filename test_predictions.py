#!/usr/bin/env python3
import requests
from PIL import Image, ImageDraw, ImageFilter
import json
import numpy as np

# Create test image 1: Normal brain (smooth, uniform)
normal_img = Image.new('RGB', (224, 224), color=(180, 180, 180))
normal_array = np.array(normal_img, dtype=float)
# Add very subtle noise
noise = np.random.normal(0, 3, (224, 224, 3))
normal_array = np.clip(normal_array + noise, 0, 255).astype(np.uint8)
normal_img = Image.fromarray(normal_array)
# Blur slightly to make it smooth (less edges)
normal_img = normal_img.filter(ImageFilter.GaussianBlur(radius=2))
normal_img.save('normal_brain.png')

# Create test image 2: Stroke with very high contrast and edges
stroke_img = Image.new('RGB', (224, 224), color=(160, 160, 160))
stroke_array = np.array(stroke_img, dtype=float)

# Add background noise
noise = np.random.normal(0, 5, (224, 224, 3))
stroke_array = np.clip(stroke_array + noise, 0, 255).astype(np.uint8)

# Create VERY dark stroke region
for i in range(70, 170):
    for j in range(70, 170):
        stroke_array[i, j] = [20, 20, 20]  # Very dark

# Add sharp, irregular edges with high-contrast spots
for _ in range(300):  # More rough spots
    x = np.random.randint(65, 175)
    y = np.random.randint(65, 175)
    stroke_array[max(0, x-3):min(224, x+4), max(0, y-3):min(224, y+4)] = [5, 5, 5]

stroke_img = Image.fromarray(stroke_array.astype(np.uint8))
stroke_img.save('stroke_brain.png')

# Create test image 3: Realistic stroke with complex texture
stroke_realistic = Image.new('RGB', (224, 224), color=(150, 150, 150))
stroke_realistic_array = np.array(stroke_realistic, dtype=float)

# Add background texture (simulating normal brain)
base_noise = np.random.normal(0, 8, (224, 224, 3))
stroke_realistic_array = np.clip(stroke_realistic_array + base_noise, 0, 255).astype(np.uint8)

# Create irregular stroke region with lots of internal variation
cx, cy = 112, 112
for i in range(224):
    for j in range(224):
        dist = np.sqrt((i - cx)**2 + (j - cy)**2)
        # Stroke region is roughly circular but irregular
        if 20 < dist < 70:
            # Stroke area: vary between dark and very dark
            if np.random.random() > 0.5:
                stroke_realistic_array[i, j] = [40, 40, 40]
            else:
                stroke_realistic_array[i, j] = [20, 20, 20]
        elif dist < 20:
            # Core - very dark
            stroke_realistic_array[i, j] = [15, 15, 15]

# Add fine texture/roughness to stroke region
for _ in range(500):
    x = np.random.randint(70, 154)
    y = np.random.randint(70, 154)
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    if dist < 70:
        # Random dark spots within stroke
        stroke_realistic_array[max(0, x-1):min(224, x+2), max(0, y-1):min(224, y+2)] = [10, 10, 10]

stroke_realistic = Image.fromarray(stroke_realistic_array.astype(np.uint8))
stroke_realistic.save('stroke_brain_extreme.png')

# Test all three images
print("=" * 70)
print("Testing Demo Prediction Logic with Multiple Images")
print("=" * 70)

tests = [
    ("NORMAL BRAIN", "normal_brain.png", "Expected: LOW stroke probability"),
    ("STROKE-LIKE", "stroke_brain.png", "Expected: MEDIUM stroke probability"),
    ("STROKE EXTREME", "stroke_brain_extreme.png", "Expected: HIGH stroke probability"),
]

for test_name, filename, expectation in tests:
    print(f"\n{test_name} ({filename})")
    print(f"  {expectation}")
    print("-" * 70)
    with open(filename, 'rb') as f:
        try:
            r = requests.post('http://localhost:5000/api/predict', files={'image': f})
            if r.status_code == 200:
                data = r.json()
                resnet = data['predictions']['resnet50']['stroke_probability']
                densenet = data['predictions']['densenet121']['stroke_probability']
                ensemble = data['predictions']['ensemble']['stroke_probability']
                
                print(f"  ResNet50:     {resnet:.4f} {'✓ STROKE' if resnet > 0.25 else '✗ NORMAL'}")
                print(f"  DenseNet121:  {densenet:.4f} {'✓ STROKE' if densenet > 0.25 else '✗ NORMAL'}")
                print(f"  Ensemble:     {ensemble:.4f} {'✓ STROKE' if ensemble > 0.25 else '✗ NORMAL'}")
            else:
                print(f"  Error: {r.status_code}")
                print(f"  {r.json()}")
        except Exception as e:
            print(f"  Request failed: {e}")

print("\n" + "=" * 70)
print("Prediction test complete!")
print("=" * 70)

