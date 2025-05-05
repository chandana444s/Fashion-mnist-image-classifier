import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# 1. Load the trained model
model = tf.keras.models.load_model('fashion_cnn_model_with_augmentation.h5')


# 2. Load the test dataset
(_, _), (x_test, y_test) = fashion_mnist.load_data()

# 3. Preprocess the data (same as in training)
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# 4. Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 5. Predict the first 5 test images
predictions = model.predict(x_test[:5])

# 6. Get the predicted class index for each image
predicted_labels = np.argmax(predictions, axis=1)

# 7. Show the image and prediction
for i in range(5):
    plt.figure(figsize=(2,2))
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {class_names[predicted_labels[i]]}\nActual: {class_names[y_test[i]]}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
