import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Avoid TensorFlow logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model once
model = tf.keras.models.load_model('fashion_cnn_model_with_augmentation.h5')

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocess image
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")  # Grayscale
    inverted = ImageOps.invert(image)
    resized = inverted.resize((28, 28))
    img_array = np.array(resized).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array, image, resized

# Streamlit UI
st.title("👗 Fashion MNIST Image Classifier")
st.write("Upload an image (preferably black clothing item on white background).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.subheader("📸 Original and Preprocessed Image")
    
    img_array, original_img, preprocessed_img = preprocess_image(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="Original Image", use_column_width=True)
    with col2:
        st.image(preprocessed_img, caption="Preprocessed Image (28x28, Inverted)", use_column_width=True)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index] * 100

    st.subheader("🎯 Prediction Result")
    st.success(f"**Predicted Class:** {class_names[predicted_index]}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Top-3 predictions
    st.subheader("🔝 Top 3 Predictions")
    top_3_indices = prediction[0].argsort()[-3:][::-1]
    top_3_classes = [class_names[i] for i in top_3_indices]
    top_3_confidences = [prediction[0][i] * 100 for i in top_3_indices]

    # Display top-3 in text
    for i in range(3):
        st.write(f"{i+1}. {top_3_classes[i]} — {top_3_confidences[i]:.2f}%")

    # Plot bar chart
    fig, ax = plt.subplots()
    bars = ax.bar(top_3_classes, top_3_confidences, color='lightblue')
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Top 3 Predictions")
    ax.set_ylim(0, 100)

    for bar, conf in zip(bars, top_3_confidences):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{conf:.1f}%", ha='center', va='bottom')

    st.pyplot(fig)
