import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf

# --- FINAL VERSION: Using the Trained CNN for Prediction ---

class AcneDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Acne Severity Classifier")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")

        self.image_path = None
        self.image_display = None
        self.original_image_pil = None # Store the original PIL image for resizing

        # --- Frames ---
        top_frame = tk.Frame(root, bg="#34495e", pady=10)
        top_frame.pack(fill=tk.X)

        main_frame = tk.Frame(root, bg="#2c3e50", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Widgets ---
        self.select_btn = tk.Button(top_frame, text="Select Image", command=self.select_image, bg="#3498db", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT, padx=10)
        self.select_btn.pack(side=tk.LEFT, padx=20)

        self.analyze_btn = tk.Button(top_frame, text="Analyze Image", command=self.analyze_image, bg="#2ecc71", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT, padx=10)
        self.analyze_btn.pack(side=tk.LEFT)

        self.image_label = tk.Label(main_frame, bg="#34495e")
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.image_label.bind('<Configure>', self.on_resize)

        self.result_text = tk.Text(main_frame, wrap=tk.WORD, bg="#34495e", fg="white", font=("Arial", 14), relief=tk.FLAT, width=40)
        self.result_text.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self.result_text.insert(tk.END, "Welcome! Initializing models, please wait...")
        self.root.update()

        # --- Initialize Models & Data ---
        self.load_models_and_data()
        
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Ready! Select an image to begin analysis.")

    def load_models_and_data(self):
        """Load all necessary models and static data at startup."""
        try:
            # --- Load Model ---
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # CHANGED: Updated path to look inside the 'models' sub-folder
            model_path = os.path.join(script_dir, '..', 'models', 'acne_transfer_model_finetuned.h5')

            if not os.path.exists(model_path):
                messagebox.showerror("Model Loading Error", f"Model file not found at the expected path: {model_path}")
                self.root.destroy()
                return
            self.cnn_model = tf.keras.models.load_model(model_path)
            
            # --- Static Data Dictionaries ---
            self.label_dict = {0: "level0", 1: "level1", 2: "level2", 3: "level3"}
            
            self.cause_dict = {
                "level0": "This appears to be clear skin or very minor.",
                "level1": "Likely due to hygiene or oily skin (Mild).",
                "level2": "May be caused by stress or poor diet (Moderate).",
                "level3": "Strongly consider consulting a dermatologist (Severe)."
            }
            
            self.tips_dict = {
                "level0": {
                    "Diet": "Maintain a balanced diet rich in fruits, vegetables, and whole grains. Stay hydrated.",
                    "Lifestyle": "Continue your current skincare routine. Ensure you get adequate sleep and manage stress."
                },
                "level1": {
                    "Diet": "Try to limit sugary foods and drinks. Incorporate more anti-inflammatory foods like green tea and berries.",
                    "Lifestyle": "Wash your face twice daily with a gentle cleanser. Avoid touching your face and change your pillowcase regularly."
                },
                "level2": {
                    "Diet": "Consider reducing dairy and high-glycemic foods (e.g., white bread, pastries). Focus on foods rich in zinc and Vitamin A/E.",
                    "Lifestyle": "Clean your phone screen often. Be consistent with a non-comedogenic skincare routine and avoid harsh scrubs."
                },
                "level3": {
                    "Diet": "While diet can help, it's not a substitute for medical treatment. Focus on a whole-foods diet.",
                    "Lifestyle": "Avoid popping or squeezing pimples to prevent scarring. Follow a dermatologist's advice for a targeted treatment plan."
                }
            }

            # --- Load MediaPipe Face Detection ---
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.1)
            
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load models or data.\nError: {e}")
            self.root.destroy()

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            self.image_path = path
            self.original_image_pil = Image.open(path)
            self.display_image()
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"Image selected: {os.path.basename(path)}\n\nClick 'Analyze Image' to proceed.")

    def on_resize(self, event):
        self.display_image()

    def display_image(self, analyzed_img_cv=None):
        if analyzed_img_cv is not None:
            img_rgb = cv2.cvtColor(analyzed_img_cv, cv2.COLOR_BGR2RGB)
            image_to_display = Image.fromarray(img_rgb)
            self.original_image_pil = image_to_display
        elif self.original_image_pil:
            image_to_display = self.original_image_pil.copy()
        else:
            return

        max_width = self.image_label.winfo_width()
        max_height = self.image_label.winfo_height()
        image_to_display.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        self.image_display = ImageTk.PhotoImage(image_to_display)
        self.image_label.config(image=self.image_display)
        self.image_label.image = self.image_display

    def analyze_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Analyzing, please wait...")
        self.root.update_idletasks()

        img = cv2.imread(self.image_path)
        if img is None:
            messagebox.showerror("Error", "Could not read the selected image.")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)
        
        annotated_image = img.copy()
        final_output = "Face not detected."

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)

            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            face_roi = img[y:y+h, x:x+w]

            if face_roi.size > 0:
                img_resized = cv2.resize(face_roi, (224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

                prediction = self.cnn_model.predict(img_preprocessed)
                class_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                class_label = self.label_dict.get(class_idx, "Unknown")
                cause_message = self.cause_dict.get(class_label, "No information available.")
                confidence_text = f"Confidence: {confidence * 100:.2f}%"
                
                tips = self.tips_dict.get(class_label, {"Diet": "N/A", "Lifestyle": "N/A"})
                diet_tip = tips["Diet"]
                lifestyle_tip = tips["Lifestyle"]

                final_output = (
                    f"--- Analysis Result ---\n"
                    f"Predicted Severity: {class_label.replace('level', 'Level ').capitalize()}\n"
                    f"Possible Cause: {cause_message}\n"
                    f"Confidence: {confidence_text}\n\n"
                    f"--- Recommendations ---\n"
                    f"Dietary Tip: {diet_tip}\n\n"
                    f"Lifestyle Tip: {lifestyle_tip}\n\n"
                    f"------------------------\n"
                    f"Disclaimer: Not a substitute for professional medical advice."
                )
        
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, final_output)
        self.display_image(analyzed_img_cv=annotated_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = AcneDetectorApp(root)
    root.mainloop()