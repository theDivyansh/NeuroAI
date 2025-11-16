import { Clock, TrendingUp, AlertCircle } from 'lucide-react';
import { Prediction } from '../lib/supabase';

interface PredictionHistoryProps {
  predictions: Prediction[];
}

export default function PredictionHistory({ predictions }: PredictionHistoryProps) {
  if (predictions.length === 0) {
    return (
      <div className="bg-white rounded-2xl p-8 shadow-lg border border-gray-200 text-center">
        <Clock className="h-16 w-16 text-gray-300 mx-auto mb-4" />
        <p className="text-gray-500 text-lg">No prediction history yet</p>
        <p className="text-gray-400 text-sm mt-2">Upload an MRI scan to get started</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-200">
      <div className="flex items-center gap-3 mb-6">
        <Clock className="h-8 w-8 text-blue-600" />
        <h3 className="text-2xl font-bold text-gray-800">Recent Predictions</h3>
        <span className="ml-auto text-sm bg-blue-100 text-blue-700 px-3 py-1 rounded-full font-semibold">
          {predictions.length} scans
        </span>
      </div>

      <div className="space-y-4 max-h-[600px] overflow-y-auto">
        {predictions.map((prediction) => (
          <div
            key={prediction.id}
            className="border border-gray-200 rounded-xl p-4 hover:shadow-md transition-all duration-200 bg-gradient-to-r from-gray-50 to-blue-50"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs text-gray-500">
                    {new Date(prediction.created_at).toLocaleString()}
                  </span>
                  {prediction.adversarial_detection.is_adversarial && (
                    <span className="text-xs bg-red-500 text-white px-2 py-0.5 rounded-full font-semibold flex items-center gap-1">
                      <AlertCircle className="h-3 w-3" />
                      Adversarial
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-3 gap-3 mb-3">
                  <div className="bg-white rounded-lg p-2">
                    <p className="text-xs text-gray-500 mb-1">ResNet50</p>
                    <p className="text-sm font-bold text-gray-900">
                      {(prediction.resnet50_prediction.stroke_probability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-white rounded-lg p-2">
                    <p className="text-xs text-gray-500 mb-1">DenseNet121</p>
                    <p className="text-sm font-bold text-gray-900">
                      {(prediction.densenet121_prediction.stroke_probability * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-white rounded-lg p-2">
                    <p className="text-xs text-gray-500 mb-1">Ensemble</p>
                    <p className="text-sm font-bold text-blue-600">
                      {(prediction.ensemble_prediction.stroke_probability * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1">
                    <TrendingUp className="h-4 w-4 text-purple-500" />
                    <span className="text-xs text-gray-600">
                      Agreement: {(prediction.model_agreement * 100).toFixed(0)}%
                    </span>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded-full font-semibold ${
                    prediction.ensemble_prediction.has_stroke
                      ? 'bg-red-100 text-red-700'
                      : 'bg-green-100 text-green-700'
                  }`}>
                    {prediction.ensemble_prediction.has_stroke ? 'Stroke' : 'Normal'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
