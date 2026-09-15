import os
import json
import numpy as np
tf = None
from utils.preprocess import preprocess_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH_H5 = os.path.join(MODEL_DIR, 'cnn_model.h5')
MODEL_PATH_KERAS = os.path.join(MODEL_DIR, 'cnn_model.keras')
MODEL_PATH_PKL = os.path.join(MODEL_DIR, 'mlp_model.pkl')
MAPPING_PATH = os.path.join(MODEL_DIR, 'class_mapping.json')

class CharacterPredictor:
    def __init__(self):
        self.model = None
        self.model_type = None # 'keras' or 'sklearn'
        self.mapping = {}
        # Lazy loading on first prediction request

    def load_model_and_mapping(self):
        """Load model weights (Keras CNN or MLP) and class mapping dictionary."""
        # Load class mapping
        if os.path.exists(MAPPING_PATH):
            with open(MAPPING_PATH, 'r') as f:
                raw_mapping = json.load(f)
                self.mapping = {int(k): v for k, v in raw_mapping.items()}
        else:
            self.mapping = {i: str(i) for i in range(10)}

        # Load model weights
        global tf
        if os.path.exists(MODEL_PATH_KERAS) or os.path.exists(MODEL_PATH_H5):
            if tf is None:
                try:
                    import tensorflow as tf
                except Exception as e:
                    print(f"[!] Keras TF import notice: {e}")

        if os.path.exists(MODEL_PATH_KERAS) and tf is not None:
            try:
                print(f"[*] Loading model from {MODEL_PATH_KERAS}")
                self.model = tf.keras.models.load_model(MODEL_PATH_KERAS)
                self.model_type = 'keras'
                return
            except Exception as e:
                print(f"[!] Keras model load notice: {e}")

        if os.path.exists(MODEL_PATH_H5):
            try:
                print(f"[*] Loading model from {MODEL_PATH_H5}")
                self.model = tf.keras.models.load_model(MODEL_PATH_H5)
                self.model_type = 'keras'
                return
            except Exception as e:
                print(f"[!] H5 model load notice: {e}")

        if os.path.exists(MODEL_PATH_PKL):
            try:
                import joblib
                print(f"[*] Loading model from {MODEL_PATH_PKL}")
                self.model = joblib.load(MODEL_PATH_PKL)
                self.model_type = 'sklearn'
                return
            except Exception as e:
                print(f"[!] Scikit-learn model load notice: {e}")

        print("[!] Model file not found. Predictor initialized in un-trained mode.")

    def predict_image(self, image_input):
        """
        Run preprocessing and model prediction on an uploaded or canvas image.
        """
        # 1. OpenCV Preprocessing Pipeline
        success, tensor_input, steps_b64, error_msg = preprocess_pipeline(image_input)
        if not success:
            return {
                "success": False,
                "error": f"Preprocessing failed: {error_msg}"
            }

        # 2. Check model readiness
        if self.model is None:
            self.load_model_and_mapping()
            if self.model is None:
                return {
                    "success": False,
                    "error": "Model not trained yet. Please run train_model.py first."
                }

        # 3. Model Inference
        if self.model_type == 'keras':
            probs = self.model.predict(tensor_input, verbose=0)[0]
        else:
            # Flatten 28x28 (1, 28, 28, 1) -> (1, 784)
            flat_input = tensor_input.reshape(1, 784)
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(flat_input)[0]
            else:
                pred = self.model.predict(flat_input)[0]
                probs = np.zeros(len(self.mapping))
                probs[pred] = 1.0

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx]) * 100.0

        predicted_char = self.mapping.get(top_idx, str(top_idx))

        # 4. Top-3 Candidate Probabilities
        top_3_indices = np.argsort(probs)[-3:][::-1]
        top_3_candidates = [
            {
                "character": self.mapping.get(int(idx), str(idx)),
                "confidence": round(float(probs[idx]) * 100.0, 2)
            }
            for idx in top_3_indices
        ]

        return {
            "success": True,
            "prediction": predicted_char,
            "confidence": round(confidence, 2),
            "top_candidates": top_3_candidates,
            "preprocessing_steps": steps_b64
        }

# Global predictor instance
predictor = CharacterPredictor()

def predict(image_input):
    return predictor.predict_image(image_input)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
        print(f"Testing prediction on {test_img}:")
        res = predict(test_img)
        print(json.dumps(res, indent=2))
