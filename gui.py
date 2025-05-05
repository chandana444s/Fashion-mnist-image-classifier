import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('fashion_cnn_model_with_augmentation.h5')

# Define the Fashion MNIST class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict_image(image_path):
    # Open and preprocess the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert to match Fashion MNIST style
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = img_array.reshape(1, 28, 28, 1)  # Add batch and channel dims

    # Get predictions from the model
    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    return class_names[predicted_class], confidence

def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Display image
    img = Image.open(file_path)
    img_resized = img.resize((150, 150))  # Resize for display
    tk_img = ImageTk.PhotoImage(img_resized)
    panel.configure(image=tk_img)
    panel.image = tk_img

    # Predict
    predicted_label, confidence = predict_image(file_path)
    result_label.config(text=f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}")

# GUI setup
root = tk.Tk()
root.title("Fashion MNIST Classifier")
root.geometry("300x400")

panel = tk.Label(root)
panel.pack(pady=10)

upload_btn = tk.Button(root, text="Upload Image", command=upload_and_predict)
upload_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 12))
result_label.pack(pady=10)

root.mainloop()
