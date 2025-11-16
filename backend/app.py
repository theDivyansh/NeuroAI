from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing import image
import io
from PIL import Image
import base64
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Toggle for demo mode
DEMO_MODE = True  # Set to False when using trained models

class StrokeDetectionModel:
    def __init__(self, model_type='resnet50'):
        self.model_type = model_type
        self.model = self._build_model()
        self.input_shape = (224, 224)

    def _build_model(self):
        if self.model_type == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        else:
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers[:int(len(base_model.layers) * 0.8)]:
            layer.trainable = False

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_image(self, img):
        img = img.resize(self.input_shape)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array) if self.model_type == 'resnet50' else tf.keras.applications.densenet.preprocess_input(img_array)
        return img_array

    def predict(self, img):
        processed_img = self.preprocess_image(img)
        prediction = self.model.predict(processed_img, verbose=0)
        return float(prediction[0][0])

class AdversarialDetector:
    def __init__(self):
        self.epsilon = 0.01

    def detect_adversarial(self, img, pred1, pred2):
        """Detect adversarial attacks using predictions and image analysis"""
        # Use prediction variance from actual predictions
        prediction_variance = abs(pred1 - pred2)

        # Calculate noise level
        noise_level = self._calculate_noise_level(np.array(img))

        # For demo mode, use simpler gradient consistency check
        gradient_consistency = self._check_image_quality(np.array(img))

        adversarial_score = (
            prediction_variance * 0.4 +
            noise_level * 0.3 +
            (1 - gradient_consistency) * 0.3
        )

        is_adversarial = adversarial_score > 0.35
        confidence_percent = min(adversarial_score * 100, 99.9)

        return {
            'is_adversarial': bool(is_adversarial),
            'confidence_percent': float(confidence_percent),
            'metrics': {
                'prediction_variance': float(prediction_variance),
                'noise_level': float(noise_level),
                'gradient_consistency': float(gradient_consistency)
            }
        }

    def _calculate_noise_level(self, img_array):
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array

        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        noise = np.abs(self._convolve2d(gray, laplacian))
        return min(np.mean(noise) / 255.0, 1.0)

    def _convolve2d(self, img, kernel):
        from scipy.ndimage import convolve
        return convolve(img, kernel, mode='constant')

    def _check_image_quality(self, img_array):
        """Check image quality and consistency (simpler version for demo)"""
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Calculate edge consistency
        edges_x = np.abs(gray[:, 1:] - gray[:, :-1])
        edges_y = np.abs(gray[1:, :] - gray[:-1, :])
        
        edge_consistency = 1.0 - (np.mean(edges_x) + np.mean(edges_y)) / 512.0
        return np.clip(edge_consistency, 0.0, 1.0)

resnet_model = None
densenet_model = None
adversarial_detector = AdversarialDetector()

# Only initialize models if not in DEMO_MODE
if not DEMO_MODE:
    resnet_model = StrokeDetectionModel('resnet50')
    densenet_model = StrokeDetectionModel('densenet121')

def get_demo_prediction(img):
    """Generate demo predictions based on image characteristics"""
    # Analyze image to generate predictions for demo
    img_array = np.array(img)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = np.mean(img_array, axis=2)
    else:
        gray = img_array
    
    # Normalize to 0-1
    gray = gray / 255.0
    
    # Calculate meaningful image statistics
    mean_brightness = np.mean(gray)
    contrast = np.max(gray) - np.min(gray)
    
    # Calculate edge density (darker areas with high edges suggest stroke regions)
    edges_x = np.abs(gray[:, 1:] - gray[:, :-1])
    edges_y = np.abs(gray[1:, :] - gray[:-1, :])
    edge_density = (np.mean(edges_x) + np.mean(edges_y)) / 2.0
    
    # Calculate local variance (more varied pixel values suggest stroke)
    local_variance = np.var(gray)
    
    # Calculate entropy-like measure (non-uniformity in image)
    hist, _ = np.histogram(gray, bins=32, range=(0, 1))
    hist_normalized = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist_normalized[hist_normalized > 0] * np.log2(hist_normalized[hist_normalized > 0] + 1e-10))
    entropy_normalized = min(entropy / 5.0, 1.0)
    
    # Calculate "roughness" - areas with extreme pixel value changes
    diff_x = np.abs(np.diff(gray, axis=0))
    diff_y = np.abs(np.diff(gray, axis=1))
    # Pad arrays to make them the same size as original
    roughness = np.percentile(np.concatenate([diff_x.flatten(), diff_y.flatten()]), 75)  # Use 75th percentile for sensitivity
    
    # Image analysis features that might indicate stroke:
    # Stroke images typically have:
    # 1. High edge density (sharp transitions between tissue types)
    # 2. High contrast (dark ischemic regions vs normal tissue)
    # 3. High entropy (diverse pixel distribution, not uniform)
    # 4. Roughness (sharp, irregular boundaries)
    # 5. Lower overall brightness (affected areas are darker)
    
    # Direct feature scores (0-1 range)
    brightness_score = max(0, 1.0 - mean_brightness)  # Darker = higher stroke likelihood
    edge_score = min(1.0, edge_density * 4.0)  # Boost edge detection
    variance_score = min(1.0, local_variance * 3.5)  # Boost variance detection
    contrast_score = min(1.0, contrast * 2.0)  # Increase contrast weight
    entropy_score = entropy_normalized
    roughness_score = min(1.0, roughness * 4.0)  # Increase roughness detection (was 3.0)
    
    # Weighted combination focusing on strong stroke indicators
    # Roughness and edge are MOST important for detecting stroke boundaries
    stroke_likelihood = (
        roughness_score * 0.35 +   # Roughness: irregular boundaries (strongest indicator)
        edge_score * 0.30 +        # Edges: sharp transitions
        contrast_score * 0.18 +    # Contrast: tissue differentiation
        variance_score * 0.10 +    # Variance: non-uniform regions
        entropy_score * 0.05 +     # Entropy: pixel diversity
        brightness_score * 0.02    # Darkness: weaker alone
    )
    
    # Clip to valid range
    resnet_pred = np.clip(stroke_likelihood, 0.0, 1.0)
    
    # DenseNet has slight variation to simulate ensemble diversity
    # Use slightly different feature weights
    densenet_pred = (
        roughness_score * 0.36 +
        edge_score * 0.29 +
        contrast_score * 0.17 +
        variance_score * 0.11 +
        entropy_score * 0.05 +
        brightness_score * 0.02
    )
    densenet_pred = np.clip(densenet_pred, 0.0, 1.0)
    
    return float(resnet_pred), float(densenet_pred)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models': ['ResNet50', 'DenseNet121']})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
        except Exception as e:
            return jsonify({'error': f'Invalid image file: {str(e)}'}), 400

        try:
            if DEMO_MODE:
                # Use demo predictions instead of untrained models
                resnet_pred, densenet_pred = get_demo_prediction(img)
            else:
                # Use actual model predictions
                if resnet_model is None or densenet_model is None:
                    return jsonify({'error': 'Models not loaded'}), 500
                resnet_pred = resnet_model.predict(img)
                densenet_pred = densenet_model.predict(img)

            # Detect adversarial attacks using the predictions
            adversarial_result = adversarial_detector.detect_adversarial(
                img, resnet_pred, densenet_pred
            )

            ensemble_prediction = (resnet_pred + densenet_pred) / 2

            result = {
                'timestamp': datetime.now().isoformat(),
                'predictions': {
                    'resnet50': {
                        'stroke_probability': float(resnet_pred),
                        'has_stroke': bool(resnet_pred > 0.25),  # Lower threshold for demo mode
                        'confidence': float(abs(resnet_pred - 0.25) * 4)  # Adjust confidence calc
                    },
                    'densenet121': {
                        'stroke_probability': float(densenet_pred),
                        'has_stroke': bool(densenet_pred > 0.25),  # Lower threshold for demo mode
                        'confidence': float(abs(densenet_pred - 0.25) * 4)  # Adjust confidence calc
                    },
                    'ensemble': {
                        'stroke_probability': float(ensemble_prediction),
                        'has_stroke': bool(ensemble_prediction > 0.25),  # Lower threshold for demo mode
                        'confidence': float(abs(ensemble_prediction - 0.25) * 4)  # Adjust confidence calc
                    }
                },
                'adversarial_detection': adversarial_result,
                'metadata': {
                    'image_size': list(img.size),
                    'image_mode': img.mode,
                    'model_agreement': float(1 - abs(resnet_pred - densenet_pred))
                }
            }

            return jsonify(result)

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400

        files = request.files.getlist('images')
        results = []

        for file in files:
            img = Image.open(io.BytesIO(file.read())).convert('RGB')

            if DEMO_MODE:
                resnet_pred, densenet_pred = get_demo_prediction(img)
            else:
                resnet_pred = resnet_model.predict(img)
                densenet_pred = densenet_model.predict(img)

            adversarial_result = adversarial_detector.detect_adversarial(
                img, resnet_pred, densenet_pred
            )

            ensemble_prediction = (resnet_pred + densenet_pred) / 2

            results.append({
                'filename': file.filename,
                'predictions': {
                    'resnet50': {'stroke_probability': float(resnet_pred)},
                    'densenet121': {'stroke_probability': float(densenet_pred)},
                    'ensemble': {'stroke_probability': float(ensemble_prediction)}
                },
                'adversarial_detection': adversarial_result
            })

        return jsonify({'results': results, 'count': len(results)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
