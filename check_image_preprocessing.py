from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('fashion_cnn_model_with_augmentation.h5')

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load and preprocess the image
img = Image.open("sample_fashion_images/Sneaker.png")
img = img.convert('L')
img = img.resize((28, 28))
img_inverted = ImageOps.invert(img)
img_array = np.array(img_inverted) / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

# Show the image
plt.imshow(img_array[0, :, :, 0], cmap="gray")
plt.title("Preprocessed Sneaker Image")
plt.axis("off")
plt.show()

# Predict
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
confidence = float(np.max(predictions))
predicted_label = class_names[predicted_index]

# Output the prediction
print(f"🎯 Prediction: {predicted_label}")
print(f"🧠 Confidence: {confidence:.2f}")

