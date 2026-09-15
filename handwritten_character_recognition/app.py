import os
import json
import base64
import io
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from predict import predictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
STATIC_METRICS_DIR = os.path.join(BASE_DIR, 'static', 'metrics')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_METRICS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render main application home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """
    Predict handwritten character from uploaded image file or interactive canvas base64 payload.
    """
    try:
        image_input = None

        # Case A: Base64 JSON payload (Canvas input)
        if request.is_json and 'image' in request.json:
            b64_data = request.json['image']
            if ',' in b64_data:
                b64_data = b64_data.split(',')[1]
            img_bytes = base64.b64decode(b64_data)
            image_input = Image.open(io.BytesIO(img_bytes))

        # Case B: File Upload (Drag-and-drop or File input)
        elif 'file' in request.files or 'image' in request.files:
            file_obj = request.files.get('file') or request.files.get('image')
            if file_obj and file_obj.filename != '':
                if not allowed_file(file_obj.filename):
                    return jsonify({
                        "success": False,
                        "error": "Invalid file format. Supported formats: PNG, JPG, JPEG, BMP, WEBP."
                    }), 400
                
                filename = secure_filename(file_obj.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_obj.save(save_path)
                image_input = save_path
            else:
                return jsonify({
                    "success": False,
                    "error": "No file uploaded."
                }), 400

        else:
            return jsonify({
                "success": False,
                "error": "No image data or file provided in request."
            }), 400

        # Execute prediction
        result = predictor.predict_image(image_input)

        if not result["success"]:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error processing request: {str(e)}"
        }), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return model training history and evaluation metrics."""
    metrics_file = os.path.join(MODEL_DIR, 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        return jsonify({"success": True, "metrics": data})
    else:
        return jsonify({
            "success": False,
            "error": "Metrics not found. Model might not have finished evaluation yet."
        })

@app.route('/train_status', methods=['GET'])
def train_status():
    """Check if trained model file exists."""
    keras_path = os.path.join(MODEL_DIR, 'cnn_model.keras')
    h5_path = os.path.join(MODEL_DIR, 'cnn_model.h5')
    is_trained = os.path.exists(keras_path) or os.path.exists(h5_path)
    return jsonify({
        "trained": is_trained,
        "model_file": 'cnn_model.keras' if os.path.exists(keras_path) else ('cnn_model.h5' if is_trained else None)
    })

if __name__ == '__main__':
    print("[*] Starting Handwritten Character Recognition Flask Web Server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
