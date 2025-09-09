import tensorflow as tf
import numpy as np

# --- IMPORTANT ---
# This script is for the TRANSFER LEARNING model.

# 1. Load the new trained model
model = tf.keras.models.load_model("acne_transfer_model_best.h5")

# 2. Load the label map
label_dict = {}
with open("labels.txt", "r") as f:
    for line in f:
        idx, label = line.strip().split(":")
        label_dict[int(idx)] = label

# Prediction function
def predict_acne_severity(image_path):
    # 3. Load the image with the correct target size for MobileNetV2
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Apply the specific preprocessing required by MobileNetV2
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_preprocessed)
    class_idx = np.argmax(prediction[0])
    class_label = label_dict[class_idx]
    confidence = np.max(prediction[0])

    # Cause dictionary (ensure keys match labels.txt)
    cause = {
        "level0": "This appears to be clear skin or very minor.",
        "level1": "Likely due to hygiene or oily skin (Mild).",
        "level2": "May be caused by stress or poor diet (Moderate).",
        "level3": "Strongly consider consulting a dermatologist (Severe)."
    }

    print(f"Prediction: {class_label.capitalize()}")
    print(f"Confidence: {confidence * 100:.2f}%")
    
    possible_cause = cause.get(class_label, "No specific cause information available.")
    print("Possible Cause:", possible_cause)

# Example usage
if __name__ == "__main__":
    try:
        image_path = input("Enter path to image: ")
        predict_acne_severity(image_path)
    except FileNotFoundError:
        print("Error: The specified image file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

