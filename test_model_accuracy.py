import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


model = tf.keras.models.load_model('fashion_cnn_model_with_augmentation.h5')

(_, _), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)


predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_labels == y_test)
print(f"✅ Model accuracy on Fashion MNIST test set: {accuracy:.2%}")

