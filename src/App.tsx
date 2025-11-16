import { useState, useEffect } from 'react';
import { Brain, Shield, Activity, Database } from 'lucide-react';
import ImageUpload from './components/ImageUpload';
import ModelResults from './components/ModelResults';
import PredictionHistory from './components/PredictionHistory';
import { savePrediction, getPredictions, Prediction } from './lib/supabase';

interface PredictionResponse {
  timestamp: string;
  predictions: {
    resnet50: {
      stroke_probability: number;
      has_stroke: boolean;
      confidence: number;
    };
    densenet121: {
      stroke_probability: number;
      has_stroke: boolean;
      confidence: number;
    };
    ensemble: {
      stroke_probability: number;
      has_stroke: boolean;
      confidence: number;
    };
  };
  adversarial_detection: {
    is_adversarial: boolean;
    confidence_percent: number;
    metrics: {
      prediction_variance: number;
      noise_level: number;
      gradient_consistency: number;
    };
  };
  metadata: {
    image_size: number[];
    image_mode: string;
    model_agreement: number;
  };
}

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<PredictionResponse | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<Prediction[]>([]);
  const [sessionId] = useState(() => `session_${Date.now()}`);

  useEffect(() => {
    loadPredictionHistory();
  }, []);

  const loadPredictionHistory = async () => {
    try {
      const history = await getPredictions(10);
      setPredictionHistory(history);
    } catch (error) {
      console.error('Error loading prediction history:', error);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedImage);

      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data: PredictionResponse = await response.json();
      setResults(data);

      await savePrediction({
        image_metadata: {
          size: data.metadata.image_size,
          mode: data.metadata.image_mode,
        },
        resnet50_prediction: data.predictions.resnet50,
        densenet121_prediction: data.predictions.densenet121,
        ensemble_prediction: data.predictions.ensemble,
        adversarial_detection: data.adversarial_detection,
        model_agreement: data.metadata.model_agreement,
        session_id: sessionId,
      });

      await loadPredictionHistory();
    } catch (error) {
      console.error('Error analyzing image:', error);
      alert('Failed to analyze image. Make sure the Flask backend is running on http://localhost:5000');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50">
      <div className="container mx-auto px-4 py-8">
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-4 mb-4">
            <Brain className="h-16 w-16 text-blue-600" />
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
              NeuroGuard AI
            </h1>
          </div>
          <p className="text-xl text-gray-600 mb-2">
            Advanced Brain MRI Stroke Detection System
          </p>
          <div className="flex items-center justify-center gap-6 text-sm text-gray-500">
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-purple-500" />
              <span>ResNet50 + DenseNet121</span>
            </div>
            <div className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-green-500" />
              <span>Adversarial Detection</span>
            </div>
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-blue-500" />
              <span>Cloud Storage</span>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          <div className="xl:col-span-2 space-y-8">
            <div className="bg-white rounded-2xl p-8 shadow-xl border border-gray-200">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-3">
                <Brain className="h-7 w-7 text-blue-600" />
                Upload MRI Scan
              </h2>
              <ImageUpload
                onImageSelect={setSelectedImage}
                selectedImage={selectedImage}
                isAnalyzing={isAnalyzing}
              />

              {selectedImage && !isAnalyzing && (
                <button
                  onClick={analyzeImage}
                  className="mt-6 w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 text-white font-bold py-4 px-8 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 text-lg"
                >
                  Analyze with AI Models
                </button>
              )}
            </div>

            {results && (
              <div className="bg-white rounded-2xl p-8 shadow-xl border border-gray-200">
                <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-3">
                  <Activity className="h-7 w-7 text-blue-600" />
                  Analysis Results
                </h2>
                <ModelResults
                  resnet50={results.predictions.resnet50}
                  densenet121={results.predictions.densenet121}
                  ensemble={results.predictions.ensemble}
                  adversarial={results.adversarial_detection}
                  modelAgreement={results.metadata.model_agreement}
                />
              </div>
            )}
          </div>

          <div className="xl:col-span-1">
            <PredictionHistory predictions={predictionHistory} />
          </div>
        </div>

        <footer className="mt-12 text-center text-gray-500 text-sm">
          <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200 max-w-4xl mx-auto">
            <p className="mb-2 font-semibold text-gray-700">
              Medical Disclaimer
            </p>
            <p className="leading-relaxed">
              This AI system is designed for research and educational purposes only.
              It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
              Always consult with qualified healthcare providers for medical decisions.
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
