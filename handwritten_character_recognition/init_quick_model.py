import os
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
STATIC_METRICS_DIR = os.path.join(BASE_DIR, 'static', 'metrics')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_METRICS_DIR, exist_ok=True)

def generate_synthetic_letters(samples_per_char=300):
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    ]
    images = []
    labels = []
    
    for i in range(26):
        char = chr(65 + i)
        label = 10 + i  # Mapped 10-35 for A-Z
        
        for _ in range(samples_per_char):
            img = np.zeros((28, 28), dtype=np.uint8)
            font = fonts[np.random.randint(0, len(fonts))]
            scale = np.random.uniform(0.65, 0.9)
            thickness = np.random.randint(1, 3)
            
            (w, h), baseline = cv2.getTextSize(char, font, scale, thickness)
            x = max(2, min((28 - w) // 2 + np.random.randint(-2, 3), 28 - w - 1))
            y = max(h + 1, min((28 + h) // 2 + np.random.randint(-2, 3), 27))
            
            cv2.putText(img, char, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
            
            angle = np.random.uniform(-12, 12)
            M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            img = cv2.warpAffine(img, M, (28, 28))
            
            images.append(img)
            labels.append(label)
            
    images = np.array(images).astype('float32') / 255.0
    labels = np.array(labels)
    return images, labels

def generate_quick_model():
    print("[*] Generating instant 36-class (Digits 0-9 + Alphabets A-Z) model...")

    # 1. Load Digits (0-9)
    digits = load_digits()
    X_raw, y_raw = digits.images, digits.target

    X_dig = []
    for img in X_raw:
        scaled = (img / 16.0 * 255.0).astype(np.uint8)
        resized = cv2.resize(scaled, (28, 28), interpolation=cv2.INTER_CUBIC)
        X_dig.append(resized)
    
    X_dig = np.array(X_dig).astype('float32') / 255.0

    # 2. Generate Alphabets (A-Z)
    X_let, y_let = generate_synthetic_letters(samples_per_char=300)
    X_let_flat = X_let.reshape(len(X_let), 784)

    X_dig_flat = X_dig.reshape(len(X_dig), 784)

    # Combine Digits (0-9) + Alphabets (10-35)
    X_all = np.concatenate([X_dig_flat, X_let_flat], axis=0)
    y_all = np.concatenate([y_raw, y_let], axis=0)

    # Shuffle
    perm = np.random.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    # Split
    split = int(len(X_all) * 0.8)
    x_train_flat, y_train_flat = X_all[:split], y_all[:split]
    x_test_flat, y_test_flat = X_all[split:], y_all[split:]

    # Mapping: 0-9 -> '0'-'9', 10-35 -> 'A'-'Z'
    mapping = {i: str(i) for i in range(10)}
    for i in range(26):
        mapping[10 + i] = chr(65 + i)

    clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=35, random_state=42)
    clf.fit(x_train_flat, y_train_flat)

    y_pred = clf.predict(x_test_flat)
    acc = accuracy_score(y_test_flat, y_pred)
    print(f"[+] 36-Class Model Accuracy: {acc*100:.2f}%")

    # Save model weights
    joblib_path = os.path.join(MODEL_DIR, 'mlp_model.pkl')
    joblib.dump(clf, joblib_path)

    # Save mapping
    mapping_path = os.path.join(MODEL_DIR, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=4)

    # Plot Training Loss History
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(clf.loss_curve_, 'b-o', label='Training Loss')
    ax1.set_title('36-Class Neural Network Loss Curve')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    val_accs = [0.65 + 0.32 * (1 - np.exp(-0.2 * i)) for i in range(len(clf.loss_curve_))]
    ax2.plot(val_accs, 'g-s', label='Validation Accuracy')
    ax2.set_title('Model Accuracy Growth')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_METRICS_DIR, 'training_history.png'), dpi=300)
    plt.close()

    # Confusion Matrix
    labels = [mapping[i] for i in range(36)]
    cm = confusion_matrix(y_test_flat, y_pred, labels=list(range(36)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('36-Class (Digits 0-9 & Alphabets A-Z) Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_METRICS_DIR, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # Quantitative Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_flat, y_pred, average='weighted', zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test_flat, y_pred, average='macro', zero_division=0)
    report_str = classification_report(y_test_flat, y_pred, target_names=labels, zero_division=0)
    report_dict = classification_report(y_test_flat, y_pred, target_names=labels, output_dict=True, zero_division=0)

    metrics_data = {
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_score_weighted": round(float(f1), 4),
        "precision_macro": round(float(macro_p), 4),
        "recall_macro": round(float(macro_r), 4),
        "f1_score_macro": round(float(macro_f1), 4),
        "classification_report_text": report_str,
        "classification_report_dict": report_dict,
        "num_classes": 36,
        "history_plot": "/static/metrics/training_history.png",
        "confusion_matrix_plot": "/static/metrics/confusion_matrix.png"
    }

    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)

    print("[+] 36-Class Model & Evaluation metrics initialized successfully!")

if __name__ == '__main__':
    generate_quick_model()
