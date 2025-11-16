# Flask Backend - NeuroGuard AI

## Backend Running Status
✅ **Port:** 5000
✅ **URL:** http://localhost:5000

---

## Available API Endpoints

### 1. **Health Check** ✅
**Purpose:** Verify if the backend server is running

- **Method:** `GET`
- **URL:** `http://localhost:5000/api/health`
- **Response:**
```json
{
  "status": "healthy",
  "models": ["ResNet50", "DenseNet121"]
}
```

---

### 2. **Single Image Prediction** 🖼️
**Purpose:** Upload a single MRI image and get stroke detection predictions

- **Method:** `POST`
- **URL:** `http://localhost:5000/api/predict`
- **Headers:** `Content-Type: multipart/form-data`
- **Body:** 
  - `image`: Image file (JPG, PNG, etc.)

**Response Example:**
```json
{
  "timestamp": "2025-11-16T11:02:10.123456",
  "predictions": {
    "resnet50": {
      "stroke_probability": 0.25,
      "has_stroke": false,
      "confidence": 0.50
    },
    "densenet121": {
      "stroke_probability": 0.28,
      "has_stroke": false,
      "confidence": 0.56
    },
    "ensemble": {
      "stroke_probability": 0.265,
      "has_stroke": false,
      "confidence": 0.53
    }
  },
  "adversarial_detection": {
    "is_adversarial": false,
    "confidence_percent": 15.2,
    "metrics": {
      "prediction_variance": 0.03,
      "noise_level": 0.12,
      "gradient_consistency": 0.85
    }
  },
  "metadata": {
    "image_size": [224, 224],
    "image_mode": "RGB",
    "model_agreement": 0.97
  }
}
```

---

### 3. **Batch Image Prediction** 📦
**Purpose:** Upload multiple MRI images and get predictions for all

- **Method:** `POST`
- **URL:** `http://localhost:5000/api/batch-predict`
- **Headers:** `Content-Type: multipart/form-data`
- **Body:**
  - `images`: Multiple image files

**Response:** Array of prediction objects

---

## How to Use

### Option 1: Via Frontend (Recommended)
1. Open `http://localhost:5176/` in your browser
2. Upload an MRI image
3. Click "Analyze"
4. View results

### Option 2: Via cURL (Command Line)

**Health Check:**
```bash
curl http://localhost:5000/api/health
```

**Single Prediction:**
```bash
curl -X POST -F "image=@/path/to/image.jpg" http://localhost:5000/api/predict
```

### Option 3: Via Python

```python
import requests

# Health Check
response = requests.get('http://localhost:5000/api/health')
print(response.json())

# Single Prediction
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/api/predict', files=files)
    print(response.json())

# Batch Prediction
files = [('images', open('img1.jpg', 'rb')), ('images', open('img2.jpg', 'rb'))]
response = requests.post('http://localhost:5000/api/batch-predict', files=files)
print(response.json())
```

---

## Configuration

### Demo Mode
The backend runs in **DEMO_MODE = True** by default, which means:
- ✅ No training data needed
- ✅ Realistic demo predictions
- ✅ 70% of images show normal results
- ✅ 20% show moderate stroke probability
- ✅ 10% show high stroke probability

To use real trained models:
1. Set `DEMO_MODE = False` in `app.py`
2. Have trained model weights available

---

## Troubleshooting

### "Not Found" Error
❌ You're accessing the wrong URL endpoint
✅ Use only: `/api/health`, `/api/predict`, or `/api/batch-predict`

### Backend Not Running
```powershell
cd "c:\Users\divya\OneDrive\Desktop\Octoberfest\nerualAI\backend"
python app.py
```

### Port Already in Use
If port 5000 is already in use, modify `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001
```

---

## Full Application URLs

| Service | URL | Status |
|---------|-----|--------|
| **Frontend** | http://localhost:5176/ | Running |
| **Backend API** | http://localhost:5000/api/ | Running |
| **Health Check** | http://localhost:5000/api/health | ✅ |

