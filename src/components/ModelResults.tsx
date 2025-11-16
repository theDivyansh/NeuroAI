import { Brain, Activity, Shield, AlertTriangle, CheckCircle, TrendingUp } from 'lucide-react';

interface PredictionResult {
  stroke_probability: number;
  has_stroke: boolean;
  confidence: number;
}

interface AdversarialResult {
  is_adversarial: boolean;
  confidence_percent: number;
  metrics: {
    prediction_variance: number;
    noise_level: number;
    gradient_consistency: number;
  };
}

interface ModelResultsProps {
  resnet50: PredictionResult;
  densenet121: PredictionResult;
  ensemble: PredictionResult;
  adversarial: AdversarialResult;
  modelAgreement: number;
}

export default function ModelResults({ resnet50, densenet121, ensemble, adversarial, modelAgreement }: ModelResultsProps) {
  const ResultCard = ({ title, icon: Icon, result, color }: { title: string; icon: any; result: PredictionResult; color: string }) => (
    <div className={`bg-gradient-to-br ${color} rounded-2xl p-6 shadow-xl border border-gray-200 transform hover:scale-105 transition-all duration-300`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <Icon className="h-8 w-8 text-gray-700" />
          <h3 className="text-xl font-bold text-gray-800">{title}</h3>
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-700">Stroke Probability</span>
            <span className="text-2xl font-bold text-gray-900">
              {(result.stroke_probability * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div
              className={`h-full transition-all duration-1000 ${
                result.has_stroke ? 'bg-red-500' : 'bg-green-500'
              }`}
              style={{ width: `${result.stroke_probability * 100}%` }}
            ></div>
          </div>
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-gray-300">
          <span className="text-sm font-medium text-gray-700">Diagnosis</span>
          <span className={`px-4 py-1 rounded-full text-sm font-bold ${
            result.has_stroke ? 'bg-red-500 text-white' : 'bg-green-500 text-white'
          }`}>
            {result.has_stroke ? 'Stroke Detected' : 'No Stroke'}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-gray-700">Confidence</span>
          <span className="text-lg font-bold text-blue-600">
            {(result.confidence * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ResultCard
          title="ResNet50"
          icon={Brain}
          result={resnet50}
          color="from-purple-100 to-pink-100"
        />
        <ResultCard
          title="DenseNet121"
          icon={Activity}
          result={densenet121}
          color="from-blue-100 to-cyan-100"
        />
      </div>

      <div className="bg-gradient-to-br from-emerald-100 to-teal-100 rounded-2xl p-8 shadow-xl border-2 border-emerald-300">
        <div className="flex items-center gap-3 mb-6">
          <TrendingUp className="h-10 w-10 text-emerald-700" />
          <h3 className="text-2xl font-bold text-gray-800">Ensemble Prediction</h3>
          <span className="ml-auto text-sm bg-emerald-600 text-white px-4 py-1 rounded-full font-semibold">
            Combined AI Models
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div className="bg-white bg-opacity-70 rounded-xl p-4 text-center">
            <p className="text-sm font-medium text-gray-600 mb-1">Stroke Probability</p>
            <p className="text-4xl font-bold text-gray-900">{(ensemble.stroke_probability * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-white bg-opacity-70 rounded-xl p-4 text-center">
            <p className="text-sm font-medium text-gray-600 mb-1">Confidence Level</p>
            <p className="text-4xl font-bold text-blue-600">{(ensemble.confidence * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-white bg-opacity-70 rounded-xl p-4 text-center">
            <p className="text-sm font-medium text-gray-600 mb-1">Model Agreement</p>
            <p className="text-4xl font-bold text-purple-600">{(modelAgreement * 100).toFixed(1)}%</p>
          </div>
        </div>

        <div className={`p-6 rounded-xl ${ensemble.has_stroke ? 'bg-red-500' : 'bg-green-500'}`}>
          <div className="flex items-center justify-center gap-3">
            {ensemble.has_stroke ? (
              <AlertTriangle className="h-8 w-8 text-white" />
            ) : (
              <CheckCircle className="h-8 w-8 text-white" />
            )}
            <p className="text-2xl font-bold text-white">
              {ensemble.has_stroke ? 'Stroke Detected - Immediate Medical Attention Required' : 'No Stroke Detected - Brain Appears Normal'}
            </p>
          </div>
        </div>
      </div>

      <div className={`rounded-2xl p-8 shadow-xl border-2 ${
        adversarial.is_adversarial
          ? 'bg-gradient-to-br from-red-100 to-orange-100 border-red-300'
          : 'bg-gradient-to-br from-green-100 to-emerald-100 border-green-300'
      }`}>
        <div className="flex items-center gap-3 mb-6">
          <Shield className={`h-10 w-10 ${adversarial.is_adversarial ? 'text-red-600' : 'text-green-600'}`} />
          <h3 className="text-2xl font-bold text-gray-800">Adversarial Attack Detection</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white bg-opacity-70 rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <p className="text-sm font-medium text-gray-700">Detection Status</p>
              <span className={`px-4 py-1 rounded-full text-sm font-bold ${
                adversarial.is_adversarial ? 'bg-red-500 text-white' : 'bg-green-500 text-white'
              }`}>
                {adversarial.is_adversarial ? 'Attack Detected' : 'Image Clean'}
              </span>
            </div>
            <p className="text-3xl font-bold text-gray-900">{adversarial.confidence_percent.toFixed(1)}%</p>
            <p className="text-sm text-gray-600 mt-2">
              {adversarial.is_adversarial ? 'Confidence of adversarial manipulation' : 'Confidence of authentic image'}
            </p>
          </div>

          <div className="bg-white bg-opacity-70 rounded-xl p-6">
            <p className="text-sm font-medium text-gray-700 mb-4">Detection Metrics</p>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-600">Prediction Variance</span>
                  <span className="font-semibold">{(adversarial.metrics.prediction_variance * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${adversarial.metrics.prediction_variance * 100}%` }}
                  ></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-600">Noise Level</span>
                  <span className="font-semibold">{(adversarial.metrics.noise_level * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-orange-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${adversarial.metrics.noise_level * 100}%` }}
                  ></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-600">Gradient Consistency</span>
                  <span className="font-semibold">{(adversarial.metrics.gradient_consistency * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-green-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${adversarial.metrics.gradient_consistency * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {adversarial.is_adversarial && (
          <div className="bg-red-500 text-white p-4 rounded-xl">
            <p className="font-semibold text-center">
              Warning: This image may have been manipulated. Verify image source before clinical use.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
