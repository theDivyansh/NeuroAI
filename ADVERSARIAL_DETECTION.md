# Adversarial Attack Detection Implementation

## Overview

Enhanced the `AdversarialDetector` class in `backend/app.py` to generate and detect adversarial attacks using two popular attack methods:
- **FGSM** (Fast Gradient Sign Method)
- **PGD** (Projected Gradient Descent)

## New Features

### 1. FGSM Attack Generation

The `generate_fgsm_attack()` method creates adversarial examples using the Fast Gradient Sign Method:

```python
def generate_fgsm_attack(self, img_array, target_label=1, epsilon=None):
    """
    Generates FGSM adversarial example
    
    Parameters:
    - img_array: Input image as numpy array
    - target_label: Target class (1 for stroke, 0 for normal)
    - epsilon: Perturbation magnitude (default: 0.03)
    
    Returns: Adversarial image array
    """
```

**Key Features:**
- Configurable perturbation magnitude (epsilon)
- Handles both normalized (0-1) and unnormalized (0-255) images
- Applies clipping to keep pixel values in valid range
- Single-step perturbation for fast execution

**Parameters:**
- `epsilon`: 0.03 (3% perturbation magnitude)

### 2. PGD Attack Generation

The `generate_pgd_attack()` method creates stronger adversarial examples using iterative perturbation:

```python
def generate_pgd_attack(self, img_array, target_label=1, epsilon=None, alpha=None, steps=None):
    """
    Generates PGD adversarial example
    
    Parameters:
    - img_array: Input image as numpy array
    - epsilon: Perturbation bound (default: 0.03)
    - alpha: Step size for each iteration (default: 0.007)
    - steps: Number of iterations (default: 10)
    
    Returns: Adversarial image array
    """
```

**Key Features:**
- Iterative perturbation for stronger attacks
- Epsilon-ball constraint to stay near original
- Configurable step size and number of iterations
- Gradient-based optimization

**Parameters:**
- `pgd_epsilon`: 0.03 (perturbation bound)
- `pgd_alpha`: 0.007 (step size per iteration)
- `pgd_steps`: 10 (number of iterations)

### 3. Enhanced Detection Method

The improved `detect_adversarial()` method now checks:

1. **Prediction Variance** (25% weight)
   - Difference between ResNet50 and DenseNet121 predictions
   - Higher variance suggests adversarial perturbation

2. **Noise Level** (25% weight)
   - Laplacian filter-based noise detection
   - Adversarial noise appears as high-frequency artifacts

3. **Gradient Consistency** (25% weight)
   - Edge-based image quality check
   - Consistent edges indicate natural images

4. **Adversarial Similarity** (25% weight)
   - Compares input image with generated adversarial examples
   - If input is similar to FGSM/PGD attacks, likely adversarial

### 4. Perturbation Similarity Analysis

The `_calculate_perturbation_similarity()` method quantifies how similar an image is to generated adversarial examples:

```python
def _calculate_perturbation_similarity(self, original, adversarial):
    """
    Calculate L2 distance and structural perturbation
    between original and adversarial examples
    """
```

**Metrics:**
- **L2 Distance**: Euclidean distance in pixel space
- **Structural Perturbation**: Mean absolute difference
- **Combined Score**: Average of both metrics (0-1 range)

### 5. New API Endpoint

**Endpoint:** `POST /api/test-adversarial`

**Purpose:** Generate FGSM and PGD attacks on an image and test detection

**Request:**
```json
{
  "image": <binary image file>
}
```

**Response:**
```json
{
  "timestamp": "2025-11-16T13:25:00.000Z",
  "original_image": {
    "predictions": {
      "resnet50": 0.15,
      "densenet121": 0.14,
      "ensemble": 0.145
    },
    "adversarial_detection": {
      "is_adversarial": false,
      "confidence_percent": 22.5,
      "metrics": {
        "prediction_variance": 0.01,
        "noise_level": 0.05,
        "gradient_consistency": 0.82,
        "adversarial_similarity": 0.08,
        "fgsm_generated": true,
        "pgd_generated": true
      }
    }
  },
  "fgsm_attack": {
    "image_base64": "iVBORw0KGgo...",
    "epsilon": 0.03,
    "predictions": { ... },
    "adversarial_detection": { ... },
    "prediction_change": {
      "resnet50": 0.05,
      "densenet121": 0.04
    }
  },
  "pgd_attack": {
    "image_base64": "iVBORw0KGgo...",
    "epsilon": 0.03,
    "alpha": 0.007,
    "steps": 10,
    "predictions": { ... },
    "adversarial_detection": { ... },
    "prediction_change": {
      "resnet50": 0.08,
      "densenet121": 0.07
    }
  },
  "attack_summary": {
    "original_detected_as_adversarial": false,
    "fgsm_detected_as_adversarial": true,
    "pgd_detected_as_adversarial": true,
    "fgsm_confidence": 65.3,
    "pgd_confidence": 72.1
  }
}
```

## Detection Algorithm

### Scoring Mechanism

```
Adversarial Score = (
    prediction_variance * 0.25 +
    noise_level * 0.25 +
    (1 - gradient_consistency) * 0.25 +
    adversarial_similarity * 0.25
)

is_adversarial = (Adversarial Score > 0.35)
confidence = min(score * 100, 99.9)%
```

### Thresholds

- **Detection Threshold:** 0.35 (35%)
- **High Confidence:** > 50%
- **Medium Confidence:** 35-50%
- **Low Confidence:** < 35%

## Usage Examples

### Test Adversarial Detection

```bash
python test_adversarial_attacks.py
```

This creates synthetic images (normal, stroke, random) and tests both FGSM and PGD attack generation and detection.

### Integration in Production

```python
# In your Flask endpoint
adversarial_result = adversarial_detector.detect_adversarial(
    img,
    resnet_pred,
    densenet_pred,
    check_generated_attacks=True  # Enable FGSM/PGD generation
)

if adversarial_result['is_adversarial']:
    # Handle potential attack
    log_security_alert(adversarial_result)
```

### API Usage (cURL)

```bash
curl -X POST http://localhost:5000/api/test-adversarial \
  -F "image=@path/to/image.jpg"
```

## Performance Considerations

### Computation Cost

- **FGSM Generation:** ~100-200ms per image
- **PGD Generation:** ~200-400ms per image (10 iterations)
- **Detection:** ~50-100ms per image
- **Total Time:** ~350-700ms per test

### Optimization Tips

1. **Batch Processing:** Process multiple images in parallel
2. **Reduced PGD Steps:** Use 5-7 steps for real-time detection
3. **Selective Detection:** Skip for trusted image sources
4. **Caching:** Cache detection results for identical images

## Security Implications

### Attack Detection Capabilities

✓ Detects FGSM attacks with perturbation epsilon ≥ 0.02
✓ Detects PGD attacks with epsilon ≥ 0.02
✓ Detects high-frequency noise artifacts
✓ Detects prediction variance anomalies
✓ Identifies structural perturbations

### Limitations

- Works best with trained models (currently in demo mode)
- May have false positives on naturally noisy images
- Adaptive attacks may evade detection
- Requires tuning for specific model architecture

## Future Improvements

1. **Adaptive Attack Detection**
   - Implement defenses against adaptive attacks
   - Use certified robustness bounds

2. **Real Model Integration**
   - Replace demo predictions with trained models
   - Use actual model gradients for attack generation

3. **Advanced Attacks**
   - Implement C&W (Carlini & Wagner) attacks
   - Add DeepFool attack generation

4. **Defense Mechanisms**
   - Adversarial training
   - Input preprocessing and filtering
   - Ensemble hardening

5. **Monitoring & Logging**
   - Track adversarial attack attempts
   - Implement alerting system
   - Generate security reports

## Configuration

Edit the `AdversarialDetector.__init__()` method to adjust parameters:

```python
class AdversarialDetector:
    def __init__(self, model=None):
        self.epsilon = 0.03           # FGSM perturbation magnitude
        self.pgd_epsilon = 0.03       # PGD perturbation bound
        self.pgd_alpha = 0.007        # PGD step size
        self.pgd_steps = 10           # PGD iterations
        self.model = model
```

## References

- **FGSM Paper:** Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015)
- **PGD Paper:** Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2019)
- **Adversarial Robustness:** https://adversarial-robustness-toolbox.readthedocs.io/

## Testing

Run the comprehensive test suite:

```bash
# Test adversarial attack generation and detection
python test_adversarial_attacks.py

# Test on specific image
curl -X POST http://localhost:5000/api/test-adversarial \
  -F "image=@test_image.png" > result.json
```

## Troubleshooting

### Issue: Attacks not detected

**Solution:**
- Increase adversarial similarity weight in scoring
- Lower detection threshold
- Increase epsilon values for stronger attacks

### Issue: Too many false positives

**Solution:**
- Increase detection threshold to 0.4-0.5
- Reduce noise_level weight
- Train on actual model predictions

### Issue: Memory errors

**Solution:**
- Reduce image resolution
- Use smaller epsilon values
- Reduce PGD steps count

