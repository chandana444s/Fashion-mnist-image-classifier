import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('fashion_cnn_model_with_augmentation.h5')


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load and preprocess image
def load_image(image_path):
    img = Image.open(image_path).convert("L")  
    img = ImageOps.invert(img)  
    img = img.resize((28, 28))  
    img_array = np.array(img).astype('float32') / 255.0  
    img_array = img_array.reshape(1, 28, 28, 1) 
    return img_array, img  

image_path = "shirt.jpg"

# Preprocess and show image
image_array, preprocessed_img = load_image(image_path)

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(Image.open(image_path), cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(preprocessed_img, cmap='gray')
plt.title("Preprocessed (28x28, Inverted)")
plt.tight_layout()
plt.show()

prediction = model.predict(image_array)
predicted_class = np.argmax(prediction)
confidence = prediction[0][predicted_class] * 100


print("====================================")
print(f"🎯 Predicted class: {class_names[predicted_class]}")
print(f"🧠 Confidence: {confidence:.2f}%")
print("====================================")
