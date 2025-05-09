import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import customtkinter as ctk
from tkinter import messagebox

# Load models
model_elbow_frac = tf.keras.models.load_model("weights/ResNet50_Elbow_frac.h5")
model_hand_frac = tf.keras.models.load_model("weights/ResNet50_Hand_frac.h5")
model_shoulder_frac = tf.keras.models.load_model("weights/ResNet50_Shoulder_frac.h5")
model_parts = tf.keras.models.load_model("weights/ResNet50_BodyParts.h5")

# Define categories
categories_parts = ["Elbow", "Hand", "Shoulder"]
categories_fracture = ['fractured', 'normal']

def predict(img, model="Parts"):
    size = 224
    if model == 'Parts':
        chosen_model = model_parts
    else:
        if model == 'Elbow':
            chosen_model = model_elbow_frac
        elif model == 'Hand':
            chosen_model = model_hand_frac
        elif model == 'Shoulder':
            chosen_model = model_shoulder_frac
        else:
            return "Unsupported model."

    temp_img = image.load_img(img, target_size=(size, size))
    x = image.img_to_array(temp_img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = np.argmax(chosen_model.predict(images), axis=1)

    if model == 'Parts':
        prediction_str = categories_parts[prediction.item()]
    else:
        prediction_str = categories_fracture[prediction.item()]

    return prediction_str

def classify_image(image_path):
    # Check if the image is a neck image or other unsupported type
    if "neck" in image_path.lower() or any(part in image_path.lower() for part in ["unsupported_part1", "unsupported_part2"]):
        return "Unsupported image type: images cannot be classified."
    
    if "ankle" in image_path.lower() or any(part in image_path.lower() for part in ["unsupported_part1", "unsupported_part2"]):
        return "Unsupported image type: images cannot be classified."
    
    if "knee" in image_path.lower() or any(part in image_path.lower() for part in ["unsupported_part1", "unsupported_part2"]):
        return "Unsupported image type: images cannot be classified."

    # Predict body part
    body_part = predict(image_path, model="Parts")
    
    # Check for unsupported body parts
    if body_part not in categories_parts:
        return "Unsupported image type: images cannot be classified."

    # Predict fracture status based on the body part
    if body_part == "Elbow":
        fracture_status = predict(image_path, model="Elbow")
    elif body_part == "Hand":
        fracture_status = predict(image_path, model="Hand")
    elif body_part == "Shoulder":
        fracture_status = predict(image_path, model="Shoulder")
    
    return f"Body Part: {body_part}, Fracture Status: {fracture_status}"

# Create a simple GUI for image upload
def upload_image():
    image_path = ctk.filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        result = classify_image(image_path)
        messagebox.showinfo("Prediction Result", result)

# GUI setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Bone Fracture Detection")
app.geometry("400x300")

upload_button = ctk.CTkButton(app, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

app.mainloop()
