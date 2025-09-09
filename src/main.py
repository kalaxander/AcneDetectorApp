import tkinter as tk
from detection import detect_acne_and_cause_from_image, detect_acne_from_webcam
from tkinter import filedialog
from PIL import Image, ImageTk

def browse_image():
    path = filedialog.askopenfilename()
    if path:
        result = detect_acne_and_cause_from_image(path)
        result_label.config(text=result)

def open_webcam():
    detect_acne_from_webcam()

app = tk.Tk()
app.title("Acne Cause Detection")
app.geometry("400x300")

tk.Label(app, text="Acne Detection Tool", font=("Helvetica", 16)).pack(pady=10)

tk.Button(app, text="Upload Image", command=browse_image).pack(pady=10)
tk.Button(app, text="Use Webcam", command=open_webcam).pack(pady=10)

result_label = tk.Label(app, text="", wraplength=350)
result_label.pack(pady=20)

def save_report():
    if result_label["text"]:
        with open("acne_report.txt", "w") as f:
            f.write(result_label["text"])
        result_label.config(text=result_label["text"] + "\n(Saved as acne_report.txt)")

save_button = tk.Button(app, text="Save Report", command=save_report)
save_button.pack(pady=10)

app.mainloop()
