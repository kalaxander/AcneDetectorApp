import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import json
from datetime import datetime
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
        self.webcam_active = False
        self.cap = None
        self.webcam_after_id = None
        self.batch_images = []
        self.batch_results = []
        self.batch_processing = False
        self.batch_cancelled = False

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

        self.webcam_btn = tk.Button(top_frame, text="Start Webcam", command=self.toggle_webcam, bg="#e74c3c", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT, padx=10)
        self.webcam_btn.pack(side=tk.LEFT, padx=(10, 0))

        self.batch_btn = tk.Button(top_frame, text="Batch Process", command=self.select_batch_images, bg="#9b59b6", fg="white", font=("Arial", 12, "bold"), relief=tk.FLAT, padx=10)
        self.batch_btn.pack(side=tk.LEFT, padx=(10, 0))

        self.image_label = tk.Label(main_frame, bg="#34495e")
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.image_label.bind('<Configure>', self.on_resize)

        # Create a notebook for tabbed interface
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Single analysis tab
        self.single_frame = tk.Frame(self.notebook, bg="#34495e")
        self.notebook.add(self.single_frame, text="Single Analysis")
        
        self.result_text = tk.Text(self.single_frame, wrap=tk.WORD, bg="#34495e", fg="white", font=("Arial", 14), relief=tk.FLAT, width=40)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Batch processing tab
        self.batch_frame = tk.Frame(self.notebook, bg="#34495e")
        self.notebook.add(self.batch_frame, text="Batch Results")
        
        # Batch controls frame
        batch_controls = tk.Frame(self.batch_frame, bg="#34495e", pady=5)
        batch_controls.pack(fill=tk.X)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(batch_controls, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_label = tk.Label(batch_controls, text="Ready for batch processing", bg="#34495e", fg="white", font=("Arial", 10))
        self.progress_label.pack(pady=2)
        
        batch_buttons = tk.Frame(batch_controls, bg="#34495e")
        batch_buttons.pack(fill=tk.X, pady=5)
        
        self.cancel_btn = tk.Button(batch_buttons, text="Cancel", command=self.cancel_batch, bg="#e74c3c", fg="white", font=("Arial", 10), relief=tk.FLAT, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_btn = tk.Button(batch_buttons, text="Export Results", command=self.export_batch_results, bg="#27ae60", fg="white", font=("Arial", 10), relief=tk.FLAT, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT, padx=5)
        
        # Batch results display
        self.batch_text = tk.Text(self.batch_frame, wrap=tk.WORD, bg="#34495e", fg="white", font=("Arial", 12), relief=tk.FLAT)
        self.batch_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.insert(tk.END, "Welcome! Initializing models, please wait...")
        self.root.update()

        # --- Initialize Models & Data ---
        self.load_models_and_data()
        
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Ready! Select an image, start webcam, or choose batch processing.")
        self.batch_text.insert(tk.END, "Select 'Batch Process' to analyze multiple images at once.")

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
            self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.3)
            
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
        # If no face detected, try with lower confidence or use whole image
        if not results.detections:
            # Try with very low confidence
            low_conf_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.1)
            results_low = low_conf_detection.process(img_rgb)
            
            if results_low.detections:
                results = results_low
            else:
                # If still no face, offer manual analysis option
                final_output = ("Face not detected automatically.\n\n"
                              "This could be due to:\n"
                              "• Partial face view\n"
                              "• Poor lighting\n"
                              "• Face angle\n\n"
                              "Try:\n"
                              "• Use a front-facing photo\n"
                              "• Ensure good lighting\n"
                              "• Click 'Manual Analysis' for custom region selection")
        
        if not results.detections:
            # Add manual analysis button
            if not hasattr(self, 'manual_btn'):
                self.manual_btn = tk.Button(self.select_btn.master, text="Manual Analysis", 
                                          command=self.manual_analysis, bg="#f39c12", fg="white", 
                                          font=("Arial", 12, "bold"), relief=tk.FLAT, padx=10)
                self.manual_btn.pack(side=tk.LEFT, padx=(10, 0))
        else:
            final_output = "Face not detected."

        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Expand bounding box by 20% to capture more facial area
            padding_x = int(w * 0.2)
            padding_y = int(h * 0.2)
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(iw - x, w + 2 * padding_x)
            h = min(ih - y, h + 2 * padding_y)

            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            face_roi = img[y:y+h, x:x+w]

            if face_roi.size > 0:
                # Ensure face ROI is valid and properly preprocessed
                img_resized = cv2.resize(face_roi, (224, 224))
                
                # Convert BGR to RGB for proper preprocessing
                img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                # Normalize pixel values to [0, 1] range
                img_array = img_resized_rgb.astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Apply MobileNetV2 preprocessing
                img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array * 255.0)

                prediction = self.cnn_model.predict(img_preprocessed, verbose=0)
                
                # Add debugging information
                print(f"Raw prediction: {prediction[0]}")
                print(f"Prediction shape: {prediction.shape}")
                
                class_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                print(f"Class index: {class_idx}, Confidence: {confidence:.4f}")
                
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
    
    def manual_analysis(self):
        """Allow manual region selection for analysis."""
        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return
        
        # Create a simple dialog for manual ROI selection
        manual_window = tk.Toplevel(self.root)
        manual_window.title("Manual Analysis - Select Face Region")
        manual_window.geometry("600x500")
        manual_window.configure(bg="#2c3e50")
        
        # Instructions
        instructions = tk.Label(manual_window, 
                               text="Click and drag to select the face region for analysis",
                               bg="#2c3e50", fg="white", font=("Arial", 12))
        instructions.pack(pady=10)
        
        # Load and display image
        img = cv2.imread(self.image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize for display
        display_img = Image.fromarray(img_rgb)
        display_img.thumbnail((500, 400), Image.Resampling.LANCZOS)
        
        self.manual_photo = ImageTk.PhotoImage(display_img)
        
        # Canvas for image and selection
        canvas = tk.Canvas(manual_window, width=500, height=400, bg="#34495e")
        canvas.pack(pady=10)
        canvas.create_image(250, 200, image=self.manual_photo)
        
        # Selection variables
        self.selection_start = None
        self.selection_rect = None
        self.manual_roi = None
        
        def start_selection(event):
            self.selection_start = (event.x, event.y)
            if self.selection_rect:
                canvas.delete(self.selection_rect)
        
        def update_selection(event):
            if self.selection_start:
                if self.selection_rect:
                    canvas.delete(self.selection_rect)
                self.selection_rect = canvas.create_rectangle(
                    self.selection_start[0], self.selection_start[1],
                    event.x, event.y, outline="red", width=2
                )
        
        def end_selection(event):
            if self.selection_start:
                # Calculate ROI coordinates
                x1, y1 = self.selection_start
                x2, y2 = event.x, event.y
                
                # Ensure proper order
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Scale to original image size
                img_h, img_w = img.shape[:2]
                display_w, display_h = display_img.size
                
                scale_x = img_w / display_w
                scale_y = img_h / display_h
                
                roi_x1 = int(x1 * scale_x)
                roi_y1 = int(y1 * scale_y)
                roi_x2 = int(x2 * scale_x)
                roi_y2 = int(y2 * scale_y)
                
                self.manual_roi = (roi_x1, roi_y1, roi_x2, roi_y2)
                print(f"ROI selected: {self.manual_roi}")
        
        canvas.bind("<Button-1>", start_selection)
        canvas.bind("<B1-Motion>", update_selection)
        canvas.bind("<ButtonRelease-1>", end_selection)
        
        # Add status label
        status_label = tk.Label(manual_window, text="No region selected", 
                               bg="#2c3e50", fg="yellow", font=("Arial", 10))
        status_label.pack(pady=5)
        
        def update_status():
            if hasattr(self, 'manual_roi') and self.manual_roi:
                x1, y1, x2, y2 = self.manual_roi
                status_label.config(text=f"Region selected: {x2-x1}x{y2-y1} pixels")
            else:
                status_label.config(text="No region selected")
            manual_window.after(100, update_status)
        
        update_status()
        
        # Analyze button
        def analyze_manual():
            print(f"Analyze button clicked. ROI: {self.manual_roi}")
            if self.manual_roi:
                x1, y1, x2, y2 = self.manual_roi
                print(f"ROI coordinates: ({x1}, {y1}) to ({x2}, {y2})")
                
                # Ensure coordinates are valid
                if x2 <= x1 or y2 <= y1:
                    messagebox.showerror("Error", "Invalid selection. Please select a proper rectangular region.")
                    return
                
                face_roi = img[y1:y2, x1:x2]
                print(f"Face ROI shape: {face_roi.shape if face_roi.size > 0 else 'Empty'}")
                
                if face_roi.size > 0:
                    try:
                        # Analyze the selected region
                        result = self.analyze_roi(face_roi)
                        
                        # Display results
                        self.result_text.delete("1.0", tk.END)
                        self.result_text.insert(tk.END, result)
                        
                        # Show analyzed image with rectangle
                        annotated_img = img.copy()
                        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        self.display_image(analyzed_img_cv=annotated_img)
                        
                        # Switch to single analysis tab to show results
                        self.notebook.select(self.single_frame)
                        
                        manual_window.destroy()
                    except Exception as e:
                        print(f"Analysis error: {e}")
                        messagebox.showerror("Analysis Error", f"Failed to analyze region: {str(e)}")
                else:
                    messagebox.showerror("Error", "Selected region is too small.")
            else:
                messagebox.showwarning("No Selection", "Please select a region first.")
        
        analyze_btn = tk.Button(manual_window, text="Analyze Selected Region", 
                               command=analyze_manual, bg="#2ecc71", fg="white", 
                               font=("Arial", 12, "bold"), relief=tk.FLAT)
        analyze_btn.pack(pady=10)
    
    def analyze_roi(self, face_roi):
        """Analyze a manually selected ROI."""
        try:
            # Ensure face ROI is valid and properly preprocessed
            img_resized = cv2.resize(face_roi, (224, 224))
            
            # Convert BGR to RGB for proper preprocessing
            img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 1] range
            img_array = img_resized_rgb.astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Apply MobileNetV2 preprocessing
            img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array * 255.0)

            prediction = self.cnn_model.predict(img_preprocessed, verbose=0)
            
            # Add debugging information
            print(f"Manual ROI - Raw prediction: {prediction[0]}")
            
            class_idx = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            print(f"Manual ROI - Class index: {class_idx}, Confidence: {confidence:.4f}")
            
            class_label = self.label_dict.get(class_idx, "Unknown")
            cause_message = self.cause_dict.get(class_label, "No information available.")
            confidence_text = f"Confidence: {confidence * 100:.2f}%"
            
            tips = self.tips_dict.get(class_label, {"Diet": "N/A", "Lifestyle": "N/A"})
            diet_tip = tips["Diet"]
            lifestyle_tip = tips["Lifestyle"]

            return (
                f"--- Manual Analysis Result ---\n"
                f"Predicted Severity: {class_label.replace('level', 'Level ').capitalize()}\n"
                f"Possible Cause: {cause_message}\n"
                f"Confidence: {confidence_text}\n\n"
                f"--- Recommendations ---\n"
                f"Dietary Tip: {diet_tip}\n\n"
                f"Lifestyle Tip: {lifestyle_tip}\n\n"
                f"------------------------\n"
                f"Note: Manual region analysis\n"
                f"Disclaimer: Not a substitute for professional medical advice."
            )
            
        except Exception as e:
            return f"Analysis error: {str(e)}"

    def toggle_webcam(self):
        """Toggle webcam on/off."""
        if not self.webcam_active:
            self.start_webcam()
        else:
            self.stop_webcam()

    def start_webcam(self):
        """Start webcam capture and real-time analysis."""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Webcam Error", "Could not access webcam. Please check if it's connected and not being used by another application.")
                return
            
            self.webcam_active = True
            self.webcam_btn.config(text="Stop Webcam", bg="#e67e22")
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "Webcam started! Real-time analysis in progress...\n\nPress 'Stop Webcam' to end session.")
            self.update_webcam_frame()
            
        except Exception as e:
            messagebox.showerror("Webcam Error", f"Failed to start webcam: {e}")

    def stop_webcam(self):
        """Stop webcam capture."""
        self.webcam_active = False
        if self.webcam_after_id:
            self.root.after_cancel(self.webcam_after_id)
            self.webcam_after_id = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.webcam_btn.config(text="Start Webcam", bg="#e74c3c")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Webcam stopped. Select an image or start webcam again to begin analysis.")
        
        # Clear the image display
        self.image_label.config(image="")
        self.image_display = None

    def update_webcam_frame(self):
        """Update webcam frame with real-time analysis."""
        if not self.webcam_active or not self.cap:
            return
        
        success, frame = self.cap.read()
        if not success:
            self.webcam_after_id = self.root.after(30, self.update_webcam_frame)
            return
        
        # Flip frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(frame_rgb)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        analysis_result = "No face detected"
        
        if results.detections:
            detection = results.detections[0]  # Use first detected face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Extract face ROI and analyze
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                try:
                    # Expand face ROI by 15% for better context
                    padding_x = int(w * 0.15)
                    padding_y = int(h * 0.15)
                    roi_x = max(0, x - padding_x)
                    roi_y = max(0, y - padding_y)
                    roi_w = min(iw - roi_x, w + 2 * padding_x)
                    roi_h = min(ih - roi_y, h + 2 * padding_y)
                    
                    expanded_roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    
                    # Ensure face ROI is valid and properly preprocessed
                    img_resized = cv2.resize(expanded_roi, (224, 224))
                    
                    # Convert BGR to RGB for proper preprocessing
                    img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    
                    # Normalize pixel values to [0, 1] range
                    img_array = img_resized_rgb.astype(np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Apply MobileNetV2 preprocessing
                    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array * 255.0)
                    
                    prediction = self.cnn_model.predict(img_preprocessed, verbose=0)
                    class_idx = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    
                    class_label = self.label_dict.get(class_idx, "Unknown")
                    cause_message = self.cause_dict.get(class_label, "No information available.")
                    
                    analysis_result = f"Severity: {class_label.replace('level', 'Level ').capitalize()}\nConfidence: {confidence * 100:.1f}%\n{cause_message}"
                    
                    # Add text overlay on frame
                    cv2.putText(annotated_frame, f"{class_label.upper()}", (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"{confidence * 100:.1f}%", (x, y + h + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                except Exception as e:
                    analysis_result = f"Analysis error: {str(e)}"
        
        # Update result text
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"--- Real-time Analysis ---\n{analysis_result}\n\n--- Instructions ---\nPosition your face in the green box for analysis.\nPress 'Stop Webcam' to end session.\n\n--- Disclaimer ---\nNot a substitute for professional medical advice.")
        
        # Display frame
        self.display_image(analyzed_img_cv=annotated_frame)
        
        # Schedule next frame update
        self.webcam_after_id = self.root.after(100, self.update_webcam_frame)  # ~10 FPS

    def select_batch_images(self):
        """Select multiple images for batch processing."""
        if self.batch_processing:
            messagebox.showwarning("Batch Processing", "Batch processing is already in progress.")
            return
            
        file_paths = filedialog.askopenfilenames(
            title="Select Images for Batch Processing",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_paths:
            self.batch_images = list(file_paths)
            self.batch_results = []
            
            self.batch_text.delete("1.0", tk.END)
            self.batch_text.insert(tk.END, f"Selected {len(self.batch_images)} images for batch processing:\n\n")
            
            for i, path in enumerate(self.batch_images, 1):
                filename = os.path.basename(path)
                self.batch_text.insert(tk.END, f"{i}. {filename}\n")
            
            self.batch_text.insert(tk.END, f"\nClick 'Start Batch Analysis' to begin processing.")
            
            # Switch to batch tab
            self.notebook.select(self.batch_frame)
            
            # Add start batch button if not exists
            if not hasattr(self, 'start_batch_btn'):
                batch_buttons = self.cancel_btn.master
                self.start_batch_btn = tk.Button(batch_buttons, text="Start Batch Analysis", command=self.start_batch_processing, bg="#3498db", fg="white", font=("Arial", 10), relief=tk.FLAT)
                self.start_batch_btn.pack(side=tk.LEFT, padx=5)
            
            self.start_batch_btn.config(state=tk.NORMAL)
    
    def start_batch_processing(self):
        """Start batch processing in a separate thread."""
        if not self.batch_images:
            messagebox.showwarning("No Images", "Please select images first.")
            return
        
        self.batch_processing = True
        self.batch_cancelled = False
        self.batch_results = []
        
        # Update UI state
        self.start_batch_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.batch_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        
        # Start processing in separate thread
        self.batch_thread = threading.Thread(target=self.process_batch_images, daemon=True)
        self.batch_thread.start()
    
    def process_batch_images(self):
        """Process all selected images in batch."""
        total_images = len(self.batch_images)
        
        for i, image_path in enumerate(self.batch_images):
            if self.batch_cancelled:
                break
                
            # Update progress
            progress = (i / total_images) * 100
            self.root.after(0, self.update_batch_progress, progress, f"Processing {os.path.basename(image_path)}...")
            
            try:
                # Process single image
                result = self.analyze_single_image_batch(image_path)
                self.batch_results.append(result)
                
                # Update results display
                self.root.after(0, self.update_batch_display, result)
                
            except Exception as e:
                error_result = {
                    'filename': os.path.basename(image_path),
                    'path': image_path,
                    'error': str(e),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.batch_results.append(error_result)
                self.root.after(0, self.update_batch_display, error_result)
        
        # Complete processing
        final_progress = 100 if not self.batch_cancelled else (len(self.batch_results) / total_images) * 100
        status = "Batch processing completed!" if not self.batch_cancelled else "Batch processing cancelled."
        self.root.after(0, self.complete_batch_processing, final_progress, status)
    
    def analyze_single_image_batch(self, image_path):
        """Analyze a single image for batch processing."""
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Could not read image file")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)
        
        result = {
            'filename': os.path.basename(image_path),
            'path': image_path,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'face_detected': False,
            'severity': 'N/A',
            'confidence': 0.0,
            'cause': 'No face detected',
            'diet_tip': 'N/A',
            'lifestyle_tip': 'N/A'
        }
        
        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            x, y, w, h = max(0, x), max(0, y), max(0, w), max(0, h)
            
            face_roi = img[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                # Ensure face ROI is valid and properly preprocessed
                img_resized = cv2.resize(face_roi, (224, 224))
                
                # Convert BGR to RGB for proper preprocessing
                img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                
                # Normalize pixel values to [0, 1] range
                img_array = img_resized_rgb.astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Apply MobileNetV2 preprocessing
                img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array * 255.0)
                
                prediction = self.cnn_model.predict(img_preprocessed, verbose=0)
                class_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                class_label = self.label_dict.get(class_idx, "Unknown")
                cause_message = self.cause_dict.get(class_label, "No information available.")
                tips = self.tips_dict.get(class_label, {"Diet": "N/A", "Lifestyle": "N/A"})
                
                result.update({
                    'face_detected': True,
                    'severity': class_label.replace('level', 'Level ').capitalize(),
                    'confidence': float(confidence),
                    'cause': cause_message,
                    'diet_tip': tips["Diet"],
                    'lifestyle_tip': tips["Lifestyle"]
                })
        
        return result
    
    def update_batch_progress(self, progress, status):
        """Update batch processing progress."""
        self.progress_var.set(progress)
        self.progress_label.config(text=status)
    
    def update_batch_display(self, result):
        """Update batch results display."""
        if 'error' in result:
            self.batch_text.insert(tk.END, f"\n❌ {result['filename']}: ERROR - {result['error']}\n")
        else:
            confidence_str = f"{result['confidence']*100:.1f}%" if result['face_detected'] else "N/A"
            self.batch_text.insert(tk.END, f"\n✅ {result['filename']}:\n")
            self.batch_text.insert(tk.END, f"   Severity: {result['severity']}\n")
            self.batch_text.insert(tk.END, f"   Confidence: {confidence_str}\n")
            self.batch_text.insert(tk.END, f"   Cause: {result['cause']}\n")
        
        self.batch_text.see(tk.END)
    
    def complete_batch_processing(self, progress, status):
        """Complete batch processing and update UI."""
        self.progress_var.set(progress)
        self.progress_label.config(text=status)
        
        self.batch_processing = False
        self.start_batch_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)
        self.batch_btn.config(state=tk.NORMAL)
        
        if self.batch_results:
            self.export_btn.config(state=tk.NORMAL)
            
            # Add summary
            total = len(self.batch_results)
            successful = sum(1 for r in self.batch_results if 'error' not in r and r['face_detected'])
            no_face = sum(1 for r in self.batch_results if 'error' not in r and not r['face_detected'])
            errors = sum(1 for r in self.batch_results if 'error' in r)
            
            self.batch_text.insert(tk.END, f"\n{'='*50}\n")
            self.batch_text.insert(tk.END, f"BATCH PROCESSING SUMMARY\n")
            self.batch_text.insert(tk.END, f"Total Images: {total}\n")
            self.batch_text.insert(tk.END, f"Successfully Analyzed: {successful}\n")
            self.batch_text.insert(tk.END, f"No Face Detected: {no_face}\n")
            self.batch_text.insert(tk.END, f"Errors: {errors}\n")
            self.batch_text.insert(tk.END, f"{'='*50}\n")
    
    def cancel_batch(self):
        """Cancel ongoing batch processing."""
        self.batch_cancelled = True
        self.progress_label.config(text="Cancelling batch processing...")
    
    def export_batch_results(self):
        """Export batch results to JSON and CSV files."""
        if not self.batch_results:
            messagebox.showwarning("No Results", "No batch results to export.")
            return
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            title="Save Batch Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("Text files", "*.txt")]
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.batch_results, f, indent=2)
                elif file_path.endswith('.csv'):
                    import csv
                    with open(file_path, 'w', newline='', encoding='utf-8') as f:
                        if self.batch_results:
                            fieldnames = self.batch_results[0].keys()
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(self.batch_results)
                else:  # .txt
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("ACNE DETECTOR BATCH ANALYSIS RESULTS\n")
                        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("="*60 + "\n\n")
                        
                        for i, result in enumerate(self.batch_results, 1):
                            f.write(f"{i}. {result['filename']}\n")
                            if 'error' in result:
                                f.write(f"   ERROR: {result['error']}\n")
                            else:
                                f.write(f"   Severity: {result['severity']}\n")
                                f.write(f"   Confidence: {result['confidence']*100:.1f}%\n")
                                f.write(f"   Cause: {result['cause']}\n")
                                f.write(f"   Diet Tip: {result['diet_tip']}\n")
                                f.write(f"   Lifestyle Tip: {result['lifestyle_tip']}\n")
                            f.write("\n")
                
                messagebox.showinfo("Export Successful", f"Results exported to: {file_path}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'batch_processing') and self.batch_processing:
            self.batch_cancelled = True

if __name__ == "__main__":
    root = tk.Tk()
    app = AcneDetectorApp(root)
    root.mainloop()