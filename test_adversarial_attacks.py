"""
Test script for adversarial attack generation and detection
Generates FGSM and PGD adversarial examples and tests detection
"""

import numpy as np
import requests
import io
from PIL import Image
import json

# Configuration
API_URL = "http://localhost:5000"
TEST_IMAGE_SIZE = (224, 224)

def create_synthetic_image(image_type='normal'):
    """Create synthetic brain scan images for testing"""
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    
    if image_type == 'normal':
        # Normal brain - smooth, uniform lighting
        img_array[:, :] = (100, 100, 100)  # Gray background
        # Add some texture variation
        for i in range(0, 210, 30):
            for j in range(0, 210, 30):
                patch = np.random.randint(90, 110, (30, 30, 3), dtype=np.uint8)
                img_array[i:i+30, j:j+30] = patch
    
    elif image_type == 'stroke':
        # Stroke - dark region with irregular edges
        img_array[:, :] = (120, 120, 120)  # Lighter base
        # Add dark stroke region
        cv, cx = 112, 112
        for i in range(50, 150):
            for j in range(50, 150):
                dist = np.sqrt((i - cv)**2 + (j - cx)**2)
                if dist < 40:
                    darkness = max(0, 50 - dist)
                    img_array[i, j] = max(0, 120 - darkness)
    
    elif image_type == 'random':
        # Random noise for testing
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    return Image.fromarray(img_array, mode='RGB')

def test_adversarial_detection(image_type='normal'):
    """Test adversarial attack generation and detection"""
    print(f"\n{'='*70}")
    print(f"Testing Adversarial Attack Detection on {image_type.upper()} image")
    print(f"{'='*70}")
    
    # Create test image
    img = create_synthetic_image(image_type)
    
    # Save image to bytes
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    # Send to adversarial test endpoint
    files = {'image': ('test_image.png', img_buffer, 'image/png')}
    
    try:
        response = requests.post(f"{API_URL}/api/test-adversarial", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Print original image analysis
            print(f"\n📊 ORIGINAL IMAGE:")
            print(f"  ResNet50 prediction:   {result['original_image']['predictions']['resnet50']:.4f}")
            print(f"  DenseNet121 prediction: {result['original_image']['predictions']['densenet121']:.4f}")
            print(f"  Ensemble prediction:    {result['original_image']['predictions']['ensemble']:.4f}")
            
            orig_detection = result['original_image']['adversarial_detection']
            print(f"\n  Adversarial Detection:")
            print(f"    Is Adversarial:      {orig_detection['is_adversarial']}")
            print(f"    Confidence:          {orig_detection['confidence_percent']:.2f}%")
            print(f"    Prediction Variance: {orig_detection['metrics']['prediction_variance']:.4f}")
            print(f"    Noise Level:         {orig_detection['metrics']['noise_level']:.4f}")
            print(f"    Gradient Consistency: {orig_detection['metrics']['gradient_consistency']:.4f}")
            print(f"    Adversarial Similarity: {orig_detection['metrics']['adversarial_similarity']:.4f}")
            
            # Print FGSM attack analysis
            print(f"\n🔴 FGSM ATTACK:")
            print(f"  Epsilon: {result['fgsm_attack']['epsilon']:.4f}")
            print(f"  Predictions:")
            print(f"    ResNet50 prediction:   {result['fgsm_attack']['predictions']['resnet50']:.4f}")
            print(f"    DenseNet121 prediction: {result['fgsm_attack']['predictions']['densenet121']:.4f}")
            print(f"    Ensemble prediction:    {result['fgsm_attack']['predictions']['ensemble']:.4f}")
            
            print(f"\n  Prediction Changes:")
            print(f"    ResNet50 change:   {result['fgsm_attack']['prediction_change']['resnet50']:.4f}")
            print(f"    DenseNet121 change: {result['fgsm_attack']['prediction_change']['densenet121']:.4f}")
            
            fgsm_detection = result['fgsm_attack']['adversarial_detection']
            print(f"\n  Adversarial Detection:")
            print(f"    Is Adversarial:      {fgsm_detection['is_adversarial']}")
            print(f"    Confidence:          {fgsm_detection['confidence_percent']:.2f}%")
            print(f"    ✓ Successfully detected!" if fgsm_detection['is_adversarial'] else "    ✗ Not detected")
            
            # Print PGD attack analysis
            print(f"\n🔵 PGD ATTACK:")
            print(f"  Epsilon: {result['pgd_attack']['epsilon']:.4f}")
            print(f"  Alpha (step size): {result['pgd_attack']['alpha']:.4f}")
            print(f"  Steps: {result['pgd_attack']['steps']}")
            print(f"  Predictions:")
            print(f"    ResNet50 prediction:   {result['pgd_attack']['predictions']['resnet50']:.4f}")
            print(f"    DenseNet121 prediction: {result['pgd_attack']['predictions']['densenet121']:.4f}")
            print(f"    Ensemble prediction:    {result['pgd_attack']['predictions']['ensemble']:.4f}")
            
            print(f"\n  Prediction Changes:")
            print(f"    ResNet50 change:   {result['pgd_attack']['prediction_change']['resnet50']:.4f}")
            print(f"    DenseNet121 change: {result['pgd_attack']['prediction_change']['densenet121']:.4f}")
            
            pgd_detection = result['pgd_attack']['adversarial_detection']
            print(f"\n  Adversarial Detection:")
            print(f"    Is Adversarial:      {pgd_detection['is_adversarial']}")
            print(f"    Confidence:          {pgd_detection['confidence_percent']:.2f}%")
            print(f"    ✓ Successfully detected!" if pgd_detection['is_adversarial'] else "    ✗ Not detected")
            
            # Summary
            print(f"\n{'─'*70}")
            print(f"🎯 ATTACK SUMMARY:")
            print(f"  Original image flagged as adversarial: {result['attack_summary']['original_detected_as_adversarial']}")
            print(f"  FGSM attack detected: {result['attack_summary']['fgsm_detected_as_adversarial']} ({result['attack_summary']['fgsm_confidence']:.2f}%)")
            print(f"  PGD attack detected: {result['attack_summary']['pgd_detected_as_adversarial']} ({result['attack_summary']['pgd_confidence']:.2f}%)")
            
            return result
        
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to Flask server at {API_URL}")
        print(f"   Make sure Flask is running: python backend/app.py")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def test_health_check():
    """Test server health"""
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        if response.status_code == 200:
            print("✓ Flask server is running and healthy")
            return True
    except:
        pass
    
    print("✗ Flask server is not responding")
    return False

if __name__ == '__main__':
    print("\n" + "="*70)
    print("ADVERSARIAL ATTACK GENERATION AND DETECTION TEST")
    print("="*70)
    
    # Check server
    if not test_health_check():
        print("\nPlease start the Flask server first:")
        print("  cd backend")
        print("  python app.py")
        exit(1)
    
    # Test with different image types
    print("\n" + "="*70)
    print("Starting adversarial attack tests...")
    print("="*70)
    
    results = {}
    for image_type in ['normal', 'stroke', 'random']:
        results[image_type] = test_adversarial_detection(image_type)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for image_type, result in results.items():
        if result:
            print(f"\n{image_type.upper()}:")
            print(f"  FGSM Detection: {'DETECTED ✓' if result['attack_summary']['fgsm_detected_as_adversarial'] else 'NOT DETECTED ✗'}")
            print(f"  PGD Detection:  {'DETECTED ✓' if result['attack_summary']['pgd_detected_as_adversarial'] else 'NOT DETECTED ✗'}")
        else:
            print(f"\n{image_type.upper()}: FAILED")
    
    print("\n" + "="*70)
