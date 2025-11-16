import requests
import json

print("Testing Final Predictions...")
print("=" * 60)

# Test normal image
r1 = requests.post('http://localhost:5000/api/predict', files={'image': open('normal_brain.png', 'rb')})
data1 = r1.json()
has_stroke_1 = data1['predictions']['ensemble']['has_stroke']
prob_1 = data1['predictions']['ensemble']['stroke_probability']
print(f"Normal Image:   has_stroke={has_stroke_1}  probability={prob_1:.4f}")

# Test stroke image  
r2 = requests.post('http://localhost:5000/api/predict', files={'image': open('stroke_brain.png', 'rb')})
data2 = r2.json()
has_stroke_2 = data2['predictions']['ensemble']['has_stroke']
prob_2 = data2['predictions']['ensemble']['stroke_probability']
print(f"Stroke Image:   has_stroke={has_stroke_2}  probability={prob_2:.4f}")

print("=" * 60)
if not has_stroke_1 and has_stroke_2:
    print("✓ CORRECT: Normal classified as NO STROKE, Stroke as STROKE")
else:
    print("✗ ISSUE: Classifications not as expected")
