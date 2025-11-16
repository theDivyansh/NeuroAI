# Adversarial Attack Detection - Visual Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                              │
│                    (Brain MRI Scan)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────▼─────────────┐
                │   Image Preprocessing    │
                │  • RGB conversion        │
                │  • Size normalization    │
                │  • Array conversion      │
                └────────────┬─────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
   │ ORIGINAL │         │  FGSM   │         │   PGD   │
   │  IMAGE   │         │ ATTACK  │         │ ATTACK  │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                   │                    │
        │              Epsilon: 0.03         Epsilon: 0.03
        │              Steps: 1           Alpha: 0.007
        │              Time: ~150ms       Steps: 10
        │                                 Time: ~300ms
        │
   ┌────┴────────────────────────────────────────┐
   │  Three parallel paths for prediction        │
   │  and adversarial analysis                   │
   └────┬────────────────────────────────────────┘
        │
    ┌───┴──────────────────────────┐
    │   Prediction Generation      │
    │                              │
    │  ┌──────────────────────┐    │
    │  │ ResNet50 Prediction  │    │
    │  └──────────────────────┘    │
    │  ┌──────────────────────┐    │
    │  │ DenseNet121 Predic.  │    │
    │  └──────────────────────┘    │
    │  ┌──────────────────────┐    │
    │  │ Ensemble Average     │    │
    │  └──────────────────────┘    │
    └───┬──────────────────────────┘
        │
        │
    ┌───┴──────────────────────────────────────────┐
    │   Four-Component Detection System            │
    │   (25% weight each)                          │
    └───┬──────────────────────────────────────────┘
        │
        ├─────────────────────────────────┐
        │                                 │
    ┌───▼────────────────┐    ┌──────────▼──────────┐
    │ PREDICTION VARIANCE │    │  NOISE LEVEL       │
    │ ┌─────────────────┐ │    │ ┌────────────────┐ │
    │ │ |Pred1 - Pred2| │ │    │ │ Laplacian      │ │
    │ │ Range: 0.0-1.0  │ │    │ │ Filter Conv    │ │
    │ │ Higher = More   │ │    │ │ Range: 0.0-1.0 │ │
    │ │ Adversarial     │ │    │ │ High = Artifact│ │
    │ └─────────────────┘ │    │ └────────────────┘ │
    └─────────────────────┘    └────────────────────┘
        │
        │
    ┌───▼────────────────┐    ┌──────────────────────┐
    │ GRADIENT CONSIST.   │    │ ADVERSARIAL SIMILAR. │
    │ ┌─────────────────┐ │    │ ┌────────────────┐   │
    │ │ Edge Quality    │ │    │ │ L2 Distance    │   │
    │ │ Analysis        │ │    │ │ to FGSM+PGD    │   │
    │ │ Range: 0.0-1.0  │ │    │ │ Range: 0.0-1.0 │   │
    │ │ Inverted Score  │ │    │ │ (NEW METRIC)   │   │
    │ └─────────────────┘ │    │ └────────────────┘   │
    └─────────────────────┘    └──────────────────────┘
        │
        └────────────────────────────┬─────────────────────────┘
                                     │
                        ┌────────────▼──────────────┐
                        │  DETECTION SCORE         │
                        │ Calculation              │
                        │                          │
                        │ Score = 0.25 × Var +     │
                        │         0.25 × Noise +   │
                        │         0.25 × (1-Cons)+ │
                        │         0.25 × Sim       │
                        └────────────┬──────────────┘
                                     │
                        ┌────────────▼──────────────┐
                        │  DECISION LOGIC          │
                        │                          │
                        │ IF score > 0.35:         │
                        │   ADVERSARIAL = TRUE     │
                        │ ELSE:                    │
                        │   ADVERSARIAL = FALSE    │
                        └────────────┬──────────────┘
                                     │
                        ┌────────────▼──────────────┐
                        │   DETECTION RESULT       │
                        │  • is_adversarial        │
                        │  • confidence %          │
                        │  • detailed metrics      │
                        │  • attack images         │
                        └──────────────────────────┘
```

---

## Data Flow Diagram

```
INPUT REQUEST
     │
     ▼
┌─────────────────┐
│  Flask Route    │ /api/test-adversarial
│  POST Handler   │
└────────┬────────┘
         │
         ▼
   ┌──────────────────┐
   │  Image Parsing   │
   │ • Validate file  │
   │ • Load from POST │
   │ • Convert RGB    │
   └────────┬─────────┘
            │
    ┌───────┴───────────────────────┐
    │                               │
    ▼                               ▼
┌──────────────────┐         ┌──────────────────┐
│ Attack Generation│         │ Prediction Gen.  │
└────┬─────────────┘         └────┬─────────────┘
     │                            │
     ├─FGSM Attack                ├─ResNet50
     │ generate_fgsm_attack()      │ get_demo_prediction()
     │ ~100-200ms                  │ ~50-100ms
     │                             │
     ├─PGD Attack                 ├─DenseNet121
     │ generate_pgd_attack()       │ get_demo_prediction()
     │ ~200-400ms                  │ ~50-100ms
     │                             │
     └──────────────┬──────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ Detection Analysis   │
         │ detect_adversarial() │
         └─────────┬────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
    ┌─FGSM─┐  ┌─PGD──┐  ┌─ORG ──┐
    │Detect│  │Detect│  │Detect │
    └──┬───┘  └──┬───┘  └───┬───┘
       │         │          │
       └─────────┼──────────┘
               │
               ▼
    ┌──────────────────────┐
    │ Response Generation  │
    │ • Predictions        │
    │ • Detections         │
    │ • Images (b64)       │
    │ • Metrics            │
    │ • Confidence         │
    └─────────┬────────────┘
              │
              ▼
    ┌──────────────────────┐
    │ JSON Response        │
    │ {                    │
    │   "timestamp": "...",│
    │   "original": {...}, │
    │   "fgsm": {...},     │
    │   "pgd": {...},      │
    │   "summary": {...}   │
    │ }                    │
    └──────────┬───────────┘
               │
               ▼
         HTTP Response (200)
```

---

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────┐
│           AdversarialDetector Class                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Configuration:                                    │
│  • epsilon = 0.03                                  │
│  • pgd_epsilon = 0.03                              │
│  • pgd_alpha = 0.007                               │
│  • pgd_steps = 10                                  │
│  • model = None                                    │
│                                                     │
│  Public Methods:                                   │
│  ┌─────────────────────────────────────┐           │
│  │ generate_fgsm_attack()              │           │
│  │ • Input: img_array                  │           │
│  │ • Output: adversarial image         │           │
│  │ • Time: ~100-200ms                  │           │
│  └─────────────────────────────────────┘           │
│                                                     │
│  ┌─────────────────────────────────────┐           │
│  │ generate_pgd_attack()               │           │
│  │ • Input: img_array                  │           │
│  │ • Output: adversarial image         │           │
│  │ • Time: ~200-400ms                  │           │
│  └─────────────────────────────────────┘           │
│                                                     │
│  ┌─────────────────────────────────────┐           │
│  │ detect_adversarial()                │           │
│  │ • Input: img, pred1, pred2          │           │
│  │ • Output: detection result          │           │
│  │ • Time: ~100-200ms                  │           │
│  └─────────────────────────────────────┘           │
│                                                     │
│  Private Methods:                                  │
│  ┌─────────────────────────────────────┐           │
│  │ _calculate_perturbation_similarity()│           │
│  │ _calculate_noise_level()            │           │
│  │ _check_image_quality()              │           │
│  │ _convolve2d()                       │           │
│  └─────────────────────────────────────┘           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Decision Tree for Adversarial Detection

```
                      INPUT IMAGE
                           │
                           ▼
              ┌─────────────────────────┐
              │ Calculate 4 Metrics:    │
              │ 1. Prediction Variance  │
              │ 2. Noise Level          │
              │ 3. Gradient Consistency │
              │ 4. Adversarial Sim.     │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Apply Weights (25% each)│
              │ and Sum Scores          │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Total Score Calculated  │
              │ (0.0 to 1.0 range)      │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         score ≤ 0.35            score > 0.35
              │                         │
              ▼                         ▼
        ┌───────────┐          ┌──────────────┐
        │ CLEAN ✓   │          │ ADVERSARIAL  │
        │ image     │          │ DETECTED ⚠️  │
        │ Confidence│          │ Confidence   │
        │ LOW 5-35% │          │ 35-100%      │
        └───────────┘          └──────────────┘
```

---

## Attack Strength Comparison

```
┌──────────────────────────────────────────────────────────┐
│           ATTACK TYPE COMPARISON                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ FGSM (Fast Gradient Sign Method)                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Strength:  ████░░░░░░ (4/10)  MEDIUM               │ │
│ │ Speed:     ██████████ (10/10) VERY FAST            │ │
│ │ Detection: ████████░░ (8/10)  HIGH                 │ │
│ │                                                     │ │
│ │ • Single-step perturbation                          │ │
│ │ • Epsilon = 0.03 (default)                          │ │
│ │ • 100-200ms execution                               │ │
│ │ • Good for quick robustness check                   │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                          │
│ PGD (Projected Gradient Descent)                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Strength:  ████████░░ (8/10)  HIGH                 │ │
│ │ Speed:     ██░░░░░░░░ (2/10)  SLOW                 │ │
│ │ Detection: ██████████ (10/10) VERY HIGH            │ │
│ │                                                     │ │
│ │ • Multi-step iterative perturbation                 │ │
│ │ • Epsilon = 0.03, Alpha = 0.007, Steps = 10        │ │
│ │ • 200-400ms execution                               │ │
│ │ • Strong attacks, harder to defend                  │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Detection Metric Weights

```
Detection Score = (Metric 1 × 0.25) + (Metric 2 × 0.25) + (Metric 3 × 0.25) + (Metric 4 × 0.25)

┌─────────────────────────┬───────────────────────────────────────────┐
│ Metric                  │ Weight & Contribution                    │
├─────────────────────────┼───────────────────────────────────────────┤
│                         │                                           │
│ 1. Prediction Variance  │ ██████░░░░ (0.25 = 25%)                 │
│    |Pred1 - Pred2|      │ Model disagreement                       │
│                         │ Higher = More adversarial               │
│                         │                                           │
│ 2. Noise Level          │ ██████░░░░ (0.25 = 25%)                 │
│    Laplacian filter     │ High-frequency artifacts                 │
│                         │ Higher = More adversarial               │
│                         │                                           │
│ 3. Gradient Consistency │ ██████░░░░ (0.25 = 25%)                 │
│    1 - edge_quality     │ Edge smoothness (inverted)              │
│                         │ Lower = More adversarial                │
│                         │                                           │
│ 4. Adversarial Simil.   │ ██████░░░░ (0.25 = 25%)  [NEW!]         │
│    vs FGSM+PGD          │ Perturbation similarity                  │
│                         │ Higher = More adversarial               │
│                         │                                           │
└─────────────────────────┴───────────────────────────────────────────┘

EQUAL WEIGHTING BENEFITS:
✓ Balanced approach
✓ No single metric dominates
✓ Comprehensive detection
✓ Flexible for different attack types
```

---

## Response Structure Hierarchy

```
{
  "timestamp": "2025-11-16T...",
  
  "original_image": {
    "predictions": {
      "resnet50": 0.15,
      "densenet121": 0.14,
      "ensemble": 0.145
    },
    "adversarial_detection": {
      "is_adversarial": false,
      "confidence_percent": 18.5,
      "metrics": {
        "prediction_variance": 0.01,
        "noise_level": 0.05,
        "gradient_consistency": 0.82,
        "adversarial_similarity": 0.08
      }
    }
  },
  
  "fgsm_attack": {
    "image_base64": "iVBORw0KGgo...",
    "epsilon": 0.03,
    "predictions": {...},
    "adversarial_detection": {...},
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
    "predictions": {...},
    "adversarial_detection": {...},
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

---

**Visual diagrams created to help understand the complete adversarial detection system architecture, data flow, and components!**

