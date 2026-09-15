import cv2
import numpy as np
import base64
from PIL import Image
import io

def convert_grayscale(img_bgr):
    """Convert BGR or BGRA image to Grayscale."""
    if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
        # Handle alpha channel
        alpha = img_bgr[:, :, 3]
        rgb = img_bgr[:, :, :3]
        # Where alpha is transparent, make background white
        white_bg = np.ones_like(rgb, dtype=np.uint8) * 255
        alpha_factor = alpha[:, :, np.newaxis] / 255.0
        composite = (rgb * alpha_factor + white_bg * (1 - alpha_factor)).astype(np.uint8)
        return cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
    elif len(img_bgr.shape) == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr.copy()

def remove_noise(img_gray):
    """Apply Gaussian blurring to smooth noise."""
    return cv2.GaussianBlur(img_gray, (3, 3), 0)

def apply_threshold(img_gray):
    """
    Apply Otsu's binarization.
    Automatically detect background vs foreground color and return inverted binary image 
    where character is WHITE (255) on a BLACK (0) background (MNIST format).
    """
    # Auto-detect if image has dark text on light background
    # Check corner pixels to determine background brightness
    h, w = img_gray.shape
    corner_pixels = np.concatenate([
        img_gray[0:max(1, h//10), 0:max(1, w//10)].ravel(),
        img_gray[0:max(1, h//10), -max(1, w//10):].ravel(),
        img_gray[-max(1, h//10):, 0:max(1, w//10)].ravel(),
        img_gray[-max(1, h//10):, -max(1, w//10):].ravel()
    ])
    mean_corner = np.mean(corner_pixels)

    if mean_corner > 127:
        # Dark text on white background -> THRESH_BINARY_INV
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Light text on dark background -> THRESH_BINARY
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def extract_and_center_character(thresh_img, target_size=(28, 28)):
    """
    Locates character contours, crops ROI with aspect-ratio preserving padding,
    and centers character in 28x28 grid.
    """
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback if no contours found (empty or solid image)
        return cv2.resize(thresh_img, target_size), thresh_img

    # Filter out tiny noise contours
    min_area = 10
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if not valid_contours:
        valid_contours = contours

    # Find combined bounding box around all valid contours
    x_min = min(cv2.boundingRect(c)[0] for c in valid_contours)
    y_min = min(cv2.boundingRect(c)[1] for c in valid_contours)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours)

    roi = thresh_img[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return cv2.resize(thresh_img, target_size), thresh_img

    # Create a square bounding area maintaining aspect ratio
    h, w = roi.shape
    max_dim = max(h, w)

    # Pad around ROI to make it square with some breathing margin
    margin = int(max_dim * 0.15)
    padded_size = max_dim + 2 * margin
    square_canvas = np.zeros((padded_size, padded_size), dtype=np.uint8)

    y_offset = (padded_size - h) // 2
    x_offset = (padded_size - w) // 2
    square_canvas[y_offset:y_offset+h, x_offset:x_offset+w] = roi

    # Resize to target (28, 28) with high-quality interpolation
    resized = cv2.resize(square_canvas, target_size, interpolation=cv2.INTER_AREA)

    return resized, roi

def normalize_image(img_28x28):
    """Normalize pixel values to [0, 1] range and reshape for CNN input."""
    normalized = img_28x28.astype('float32') / 255.0
    return np.expand_dims(normalized, axis=(0, -1)) # Shape: (1, 28, 28, 1)

def encode_img_to_base64(img_array):
    """Convert numpy array image to base64 data URI string for frontend display."""
    if len(img_array.shape) == 2:
        pil_img = Image.fromarray(img_array)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def preprocess_pipeline(image_input):
    """
    Executes complete OpenCV preprocessing pipeline.
    
    Args:
        image_input: Can be a file path (str), numpy array (BGR), or PIL Image.
        
    Returns:
        tuple: (success, processed_tensor, steps_dict, error_msg)
    """
    try:
        # Load / convert input image to numpy array BGR format
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
            if img_bgr is None:
                raise ValueError("Could not read image from path.")
        elif isinstance(image_input, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
        else:
            raise ValueError("Unsupported image input format.")

        # Step 1: Original image snapshot
        step_original = img_bgr.copy()

        # Step 2: Grayscale conversion
        step_gray = convert_grayscale(img_bgr)

        # Step 3: Denoising & Blurring
        step_denoised = remove_noise(step_gray)

        # Step 4: Otsu Binarization / Thresholding (Inverted background)
        step_thresh = apply_threshold(step_denoised)

        # Step 5: Extract ROI and center on 28x28 grid
        final_28x28, step_roi = extract_and_center_character(step_thresh)

        # Step 6: Normalize for Keras CNN model input
        tensor_input = normalize_image(final_28x28)

        # Pack intermediate steps as base64 images for visual pipeline breakdown
        steps_b64 = {
            "original": encode_img_to_base64(step_original),
            "grayscale": encode_img_to_base64(step_gray),
            "threshold": encode_img_to_base64(step_thresh),
            "roi_crop": encode_img_to_base64(step_roi),
            "processed_28x28": encode_img_to_base64(final_28x28)
        }

        return True, tensor_input, steps_b64, None

    except Exception as e:
        return False, None, None, str(e)
