import os
import json
import ssl
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.datasets import mnist
except Exception as e:
    tf = None
    print(f"[!] Keras import notice: {e}")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
STATIC_METRICS_DIR = os.path.join(BASE_DIR, 'static', 'metrics')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_METRICS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

def generate_synthetic_letters(samples_per_char=400):
    """Generate synthetic letter samples (A-Z, mapped to labels 10-35)."""
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    ]
    images = []
    labels = []
    
    for i in range(26):
        char = chr(65 + i)
        label = 10 + i  # Class 10 to 35 for A-Z
        
        for _ in range(samples_per_char):
            img = np.zeros((28, 28), dtype=np.uint8)
            font = fonts[np.random.randint(0, len(fonts))]
            scale = np.random.uniform(0.65, 0.9)
            thickness = np.random.randint(1, 3)
            
            (w, h), baseline = cv2.getTextSize(char, font, scale, thickness)
            x = max(2, min((28 - w) // 2 + np.random.randint(-2, 3), 28 - w - 1))
            y = max(h + 1, min((28 + h) // 2 + np.random.randint(-2, 3), 27))
            
            cv2.putText(img, char, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
            
            # Apply slight rotation / shear distortion
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            img = cv2.warpAffine(img, M, (28, 28))
            
            images.append(img)
            labels.append(label)
            
    images = np.array(images).astype('float32') / 255.0
    images = np.expand_dims(images, -1)
    labels = np.array(labels)
    return images, labels

def load_data(mode='combined'):
    """
    Load dataset for training:
    - 'mnist': Digits 0-9 (10 classes)
    - 'emnist': Alphabets A-Z (26 classes)
    - 'combined': Digits (0-9) + Alphabets (A-Z) (36 classes total)
    """
    print(f"[*] Loading dataset mode: {mode}")

    # 1. Load Digits (0-9)
    try:
        (x_train_num, y_train_num), (x_test_num, y_test_num) = mnist.load_data()
        x_train_num = x_train_num.astype('float32') / 255.0
        x_test_num = x_test_num.astype('float32') / 255.0
        x_train_num = np.expand_dims(x_train_num, -1)
        x_test_num = np.expand_dims(x_test_num, -1)
    except Exception as e:
        print(f"[!] MNIST load notice: {e}. Using builtin digits dataset...")
        from sklearn.datasets import load_digits
        digits = load_digits()
        X_raw, y_raw = digits.images, digits.target
        X_28 = []
        for img in X_raw:
            scaled = (img / 16.0 * 255.0).astype(np.uint8)
            resized = cv2.resize(scaled, (28, 28), interpolation=cv2.INTER_CUBIC)
            X_28.append(resized)
        X_28 = np.array(X_28).astype('float32') / 255.0
        X_28 = np.expand_dims(X_28, -1)
        split = int(len(X_28) * 0.8)
        x_train_num, y_train_num = X_28[:split], y_raw[:split]
        x_test_num, y_test_num = X_28[split:], y_raw[split:]

    if mode == 'mnist':
        mapping = {i: str(i) for i in range(10)}
        return (x_train_num, y_train_num), (x_test_num, y_test_num), mapping

    # 2. Load Alphabets (A-Z)
    x_train_let, y_train_let = None, None
    try:
        from emnist import extract_training_samples, extract_test_samples
        print("[*] EMNIST library available. Extracting EMNIST letters...")
        x_train_let, y_train_let = extract_training_samples('letters')
        x_test_let, y_test_let = extract_test_samples('letters')
        x_train_let = x_train_let.astype('float32') / 255.0
        x_test_let = x_test_let.astype('float32') / 255.0
        x_train_let = np.expand_dims(x_train_let, -1)
        x_test_let = np.expand_dims(x_test_let, -1)
        # Shift EMNIST letters 1-26 to 0-25
        y_train_let = y_train_let - 1
        y_test_let = y_test_let - 1
    except Exception as e:
        print(f"[!] EMNIST extract notice: {e}. Generating synthetic alphabet dataset for A-Z...")
        x_let, y_let = generate_synthetic_letters(samples_per_char=500)
        split = int(len(x_let) * 0.8)
        x_train_let, y_train_let = x_let[:split], y_let[:split] - 10 # 0-25
        x_test_let, y_test_let = x_let[split:], y_let[split:] - 10

    if mode == 'emnist':
        mapping = {i: chr(65 + i) for i in range(26)}
        return (x_train_let, y_train_let), (x_test_let, y_test_let), mapping

    # Mode: 'combined' (36 classes: 0-9 and A-Z)
    print("[*] Building combined 36-class dataset (Digits 0-9 + Alphabets A-Z)...")
    y_train_let_offset = y_train_let + 10 # 10-35
    y_test_let_offset = y_test_let + 10

    x_train = np.concatenate([x_train_num, x_train_let], axis=0)
    y_train = np.concatenate([y_train_num, y_train_let_offset], axis=0)
    x_test = np.concatenate([x_test_num, x_test_let], axis=0)
    y_test = np.concatenate([y_test_num, y_test_let_offset], axis=0)

    # Shuffle
    shuffle_idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[shuffle_idx], y_train[shuffle_idx]

    # Create mapping: 0-9 -> '0'-'9', 10-35 -> 'A'-'Z'
    mapping = {i: str(i) for i in range(10)}
    for i in range(26):
        mapping[10 + i] = chr(65 + i)

    return (x_train, y_train), (x_test, y_test), mapping

def build_cnn_model(num_classes):
    """
    Build Convolutional Neural Network architecture.
    """
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_and_save_metrics(history, y_true, y_pred, mapping):
    """
    Generate and save evaluation plots and metrics JSON:
    - Training & validation accuracy/loss graphs
    - Confusion matrix heatmap
    - Precision, Recall, F1 Score & Classification Report JSON
    """
    # 1. Training History Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    ax1.plot(epochs, history.history['accuracy'], 'b-o', label='Training Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], 'r-s', label='Validation Accuracy')
    ax1.set_title('CNN Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history.history['loss'], 'b-o', label='Training Loss')
    ax2.plot(epochs, history.history['val_loss'], 'r-s', label='Validation Loss')
    ax2.set_title('CNN Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    history_plot_path = os.path.join(STATIC_METRICS_DIR, 'training_history.png')
    plt.savefig(history_plot_path, dpi=300)
    plt.close()

    # 2. Confusion Matrix Plot
    num_classes = len(mapping)
    labels = [mapping[i] for i in range(num_classes)]
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8) if num_classes > 15 else (7, 6))
    sns.heatmap(cm, annot=(num_classes <= 15), fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_plot_path = os.path.join(STATIC_METRICS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    # 3. Quantitative Metrics (Accuracy, Precision, Recall, F1)
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    report_str = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)

    metrics_data = {
        "accuracy": round(acc, 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_score_weighted": round(float(f1), 4),
        "precision_macro": round(float(macro_p), 4),
        "recall_macro": round(float(macro_r), 4),
        "f1_score_macro": round(float(macro_f1), 4),
        "classification_report_text": report_str,
        "classification_report_dict": report_dict,
        "num_classes": num_classes,
        "history_plot": "/static/metrics/training_history.png",
        "confusion_matrix_plot": "/static/metrics/confusion_matrix.png"
    }

    metrics_json_path = os.path.join(MODEL_DIR, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)

    print(f"[+] Evaluation Metrics saved successfully to {metrics_json_path}")
    print(f"    Accuracy: {acc*100:.2f}% | F1-Score: {f1*100:.2f}%")

def train(epochs=10, batch_size=128, mode='combined'):
    """
    Train CNN model and save weights & evaluation artifacts.
    """
    (x_train, y_train), (x_test, y_test), mapping = load_data(mode=mode)
    num_classes = len(mapping)

    # Save class mapping
    mapping_path = os.path.join(MODEL_DIR, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=4)

    print(f"[*] Building CNN model for {num_classes} classes...")
    model = build_cnn_model(num_classes)
    model.summary()

    model_save_h5 = os.path.join(MODEL_DIR, 'cnn_model.h5')
    model_save_keras = os.path.join(MODEL_DIR, 'cnn_model.keras')

    callbacks = [
        ModelCheckpoint(model_save_keras, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]

    print(f"[*] Training CNN model for {epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    # Save final model in both .h5 and .keras formats
    model.save(model_save_h5)
    print(f"[+] Model saved as cnn_model.h5 and cnn_model.keras")

    # Evaluate on test set
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    plot_and_save_metrics(history, y_test, y_pred, mapping)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train CNN model for Handwritten Character Recognition")
    parser.add_argument('--epochs', type=int, default=8, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--mode', type=str, default='mnist', choices=['mnist', 'emnist', 'combined'], help="Dataset mode")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, mode=args.mode)
