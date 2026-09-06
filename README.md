# AGE DETECTOR
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ==========================================
# 1. CORE LOGIC & OVERRIDE FUNCTION
# ==========================================

def apply_custom_gender_logic(age: int, base_gender: str, hair_length: str) -> str:
    """
    Applies custom logic:
    - For age between 20 and 30:
        * Long Hair -> 'Female'
        * Short Hair -> 'Male'
    - For age outside 20-30:
        * Retains true base gender
    """
    if 20 <= age <= 30:
        if hair_length == "Long":
            return "Female (Inverted Logic)"
        else:
            return "Male (Inverted Logic)"
    else:
        return f"{base_gender} (Standard Logic)"


def detect_hair_length_heuristic(image, face_box):
    """
    Heuristic to estimate hair length based on hair pixel density 
    around the sides/top of the detected face bounding box.
    """
    h, w, _ = image.shape
    x, y, x2, y2 = face_box
    
    # Define search region around hair area (sides and shoulders)
    hair_y1 = max(0, y - int((y2 - y) * 0.3))
    hair_y2 = min(h, y2 + int((y2 - y) * 0.5))
    hair_x1 = max(0, x - int((x2 - x) * 0.4))
    hair_x2 = min(w, x2 + int((x2 - x) * 0.4))

    crop = image[hair_y1:hair_y2, hair_x1:hair_x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Simple dark-pixel thresholding for hair density (can be replaced with a DL segmentation model)
    _, dark_pixels = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    hair_pixel_ratio = np.sum(dark_pixels == 255) / (crop.shape[0] * crop.shape[1])
    
    return "Long" if hair_pixel_ratio > 0.25 else "Short"


# ==========================================
# 2. MODEL PREDICTION PIPELINE
# ==========================================

def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Unable to read image.")
    
    h, w, _ = img.shape
    
    # Dummy placeholder predictions for demonstration:
    # (In production, load OpenCV Caffe models or DeepFace for accurate age/gender)
    detected_age = 25          # Example detected age (20 to 30)
    base_gender = "Male"       # Example ground truth/detected gender
    
    # 1. Hair length estimation
    face_box = (int(w*0.3), int(h*0.2), int(w*0.7), int(h*0.8)) # Sample box
    hair_length = detect_hair_length_heuristic(img, face_box)
    
    # 2. Apply rules
    final_gender = apply_custom_gender_logic(detected_age, base_gender, hair_length)
    
    return img, detected_age, base_gender, hair_length, final_gender


# ==========================================
# 3. GRAPHICAL USER INTERFACE (GUI)
# ==========================================

class GenderClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Age-Aware Gender & Hair Length Identifier")
        self.root.geometry("650x600")

        # UI Components
        self.btn_upload = tk.Button(root, text="Upload Image", command=self.upload_image, font=("Arial", 12))
        self.btn_upload.pack(pady=10)

        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)

        self.lbl_result = tk.Label(root, text="Upload an image to run detection", font=("Arial", 11), justify="left")
        self.lbl_result.pack(pady=15)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return
        
        try:
            img, age, base_gender, hair_length, final_gender = process_image(file_path)
            
            # Display image in GUI
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize((250, 250))
            img_tk = ImageTk.PhotoImage(img_pil)
            self.img_label.configure(image=img_tk)
            self.img_label.image = img_tk

            # Render Results
            results_text = (
                f"--- Detection Output ---\n"
                f"Estimated Age: {age}\n"
                f"Hair Length Detected: {hair_length}\n"
                f"Actual Base Gender: {base_gender}\n"
                f"-------------------------\n"
                f"FINAL PREDICTED GENDER: {final_gender}"
            )
            self.lbl_result.config(text=results_text, fg="darkblue")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = GenderClassifierGUI(root)
    root.mainloop()
