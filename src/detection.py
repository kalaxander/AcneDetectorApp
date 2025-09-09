import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# --- METHOD 2: Using the Trained CNN Model ---

# 1. Load your trained transfer learning model
try:
    model = tf.keras.models.load_model('acne_transfer_model_best.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'acne_transfer_model_best.h5' is in the correct directory.")
    exit()

# 2. Load the label map
label_dict = {}
try:
    with open("labels.txt", "r") as f:
        for line in f:
            idx, label = line.strip().split(":")
            label_dict[int(idx)] = label
except FileNotFoundError:
    print("Error: labels.txt not found. Please ensure it is in the correct directory.")
    exit()

# 3. Define cause dictionary (ensure keys match labels.txt)
cause_dict = {
    "level0": "This appears to be clear skin or very minor.",
    "level1": "Likely due to hygiene or oily skin (Mild).",
    "level2": "May be caused by stress or poor diet (Moderate).",
    "level3": "Strongly consider consulting a dermatologist (Severe)."
}

# --- Initialization ---
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find faces
    results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)

            # Ensure the bounding box is within the image dimensions
            x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)

            # --- PREDICTION LOGIC ---
            # 1. Extract the face Region of Interest (ROI)
            face_roi = image[y:y+h, x:x+w]

            # Check if the ROI is valid
            if face_roi.size == 0:
                continue

            # 2. Preprocess the ROI for the model
            img_resized = cv2.resize(face_roi, (224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            # 3. Make a prediction
            prediction = model.predict(img_preprocessed)
            class_idx = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            # 4. Get the label and cause
            class_label = label_dict.get(class_idx, "Unknown")
            message = cause_dict.get(class_label, "No information available.")
            confidence_text = f"Confidence: {confidence * 100:.2f}%"

            # Draw bounding box and display results
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, message, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, confidence_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display disclaimer
    cv2.putText(image, "Disclaimer: Not medical advice.", (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Show the final output
    cv2.imshow('Real-Time Acne Severity Classification', image)

    # Exit on 'q' key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
