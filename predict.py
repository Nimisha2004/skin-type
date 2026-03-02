import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, filedialog

# ------------------------
# Load trained model
# ------------------------
model = tf.keras.models.load_model("skin_type_model_v2.h5")
labels = ["oily", "normal", "dry"]

# ------------------------
# Test-Time Augmentation (TTA) function
# ------------------------
def tta_predict(image, model, num_augments=5):
    """
    Predicts skin type using multiple augmented versions of the image
    and averages the results for better robustness.
    """
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = []

    # Original
    preds.append(model.predict(img, verbose=0))

    # Augmentations
    for _ in range(num_augments):
        aug_img = img.copy()

        # Horizontal flip
        if np.random.rand() > 0.5:
            aug_img = aug_img[:, :, ::-1, :]

        # Random brightness
        factor = np.random.uniform(0.7, 1.3)
        aug_img = np.clip(aug_img * factor, 0, 1)

        # Random zoom
        zoom = np.random.uniform(0.9, 1.1)
        h, w = aug_img.shape[1:3]
        ch = int(h * zoom)
        cw = int(w * zoom)
        start_h = max(0, (h - ch)//2)
        start_w = max(0, (w - cw)//2)
        aug_img_cropped = aug_img[:, start_h:start_h+ch, start_w:start_w+cw, :]
        aug_img_resized = tf.image.resize(aug_img_cropped, (224, 224))
        preds.append(model.predict(aug_img_resized, verbose=0))

    # Average predictions
    avg_preds = np.mean(np.array(preds), axis=0)
    return avg_preds

# ------------------------
# Prediction function
# ------------------------
def predict_skin(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image.")
        return

    # Use the full image (no face detector needed)
    preds = tta_predict(img, model, num_augments=5)
    idx = np.argmax(preds)
    label = labels[idx]
    confidence = preds[0][idx] * 100

    print(f"Predicted: {label} ({confidence:.2f}% confidence)")

    # Draw rectangle around whole image and label
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w-1, h-1), (0, 255, 0), 2)
    cv2.putText(img, f"{label} {confidence:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Prediction Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------
# File picker
# ------------------------
root = Tk()
root.withdraw()  # hide main window

file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
)

if file_path:
    predict_skin(file_path)
else:
    print("No file selected.")