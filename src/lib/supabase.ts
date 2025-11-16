import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

let supabase: any = null;

if (supabaseUrl && supabaseAnonKey) {
  supabase = createClient(supabaseUrl, supabaseAnonKey);
} else {
  console.warn(
    "Supabase environment variables not configured. Using localStorage fallback."
  );
}

export interface Prediction {
  id: string;
  created_at: string;
  image_metadata: {
    size: number[];
    mode: string;
  };
  resnet50_prediction: {
    stroke_probability: number;
    has_stroke: boolean;
    confidence: number;
  };
  densenet121_prediction: {
    stroke_probability: number;
    has_stroke: boolean;
    confidence: number;
  };
  ensemble_prediction: {
    stroke_probability: number;
    has_stroke: boolean;
    confidence: number;
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
  model_agreement: number;
  session_id?: string;
}

export async function savePrediction(
  predictionData: Omit<Prediction, "id" | "created_at">
) {
  if (supabase) {
    const { data, error } = await supabase
      .from("predictions")
      .insert([predictionData])
      .select()
      .maybeSingle();

    if (error) throw error;
    return data;
  } else {
    // Fallback to localStorage
    const newPrediction: Prediction = {
      id: `pred_${Date.now()}`,
      created_at: new Date().toISOString(),
      ...predictionData,
    };

    const predictions = JSON.parse(
      localStorage.getItem("neuroguard_predictions") || "[]"
    ) as Prediction[];
    predictions.unshift(newPrediction);
    localStorage.setItem("neuroguard_predictions", JSON.stringify(predictions));
    return newPrediction;
  }
}

export async function getPredictions(limit = 10) {
  if (supabase) {
    const { data, error } = await supabase
      .from("predictions")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(limit);

    if (error) throw error;
    return data as Prediction[];
  } else {
    // Fallback to localStorage
    const predictions = JSON.parse(
      localStorage.getItem("neuroguard_predictions") || "[]"
    ) as Prediction[];
    return predictions.slice(0, limit);
  }
}

export async function getSessionPredictions(sessionId: string) {
  if (supabase) {
    const { data, error } = await supabase
      .from("predictions")
      .select("*")
      .eq("session_id", sessionId)
      .order("created_at", { ascending: false });

    if (error) throw error;
    return data as Prediction[];
  } else {
    // Fallback to localStorage
    const predictions = JSON.parse(
      localStorage.getItem("neuroguard_predictions") || "[]"
    ) as Prediction[];
    return predictions.filter((p) => p.session_id === sessionId);
  }
}
