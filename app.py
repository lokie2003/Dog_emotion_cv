import streamlit as st
import numpy as np
import tensorflow as tf
import pathlib
import tensorflow_hub as hub
import cv2  # Import OpenCV
from PIL import Image

# Register the custom KerasLayer
hub_layer = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4", trainable=False)

# Define the emotion labels
emotion_labels = ['angry', 'happy', 'relaxed', 'sad']

# Load the pre-trained model
def load_model():
    with tf.keras.utils.custom_object_scope({'KerasLayer': hub_layer}):
        model = tf.keras.models.load_model('dog_emotion.h5')
    return model

model = load_model()

# Define a function to preprocess the image using OpenCV
def preprocess_image_cv2(image):
    # Resize the image to 224x224 pixels
    img = cv2.resize(image, (224, 224))
    img = img / 255.0  # Normalize the image
    return img

# Streamlit UI
st.title('DOG EMOTION CLASSIFIER')
# Add borders to separate tabs
st.markdown(
    "<style>"
    ".stSelectbox { border: 2px solid #ccc; border-radius: 4px; padding: 8px; }"
    "</style>",
    unsafe_allow_html=True,
)
# Create tabs using selectbox
selected_tab = st.selectbox("SELECT A TAB", ["INTRODUCTION", "PREDICTION"])

# Introduction tab
if selected_tab == "INTRODUCTION":
    st.title('INTRODUCTION')
    st.write('THIS IS DOG EMOTION CLASSIFIER USING RESNET FEATURE ENGINEERING')
    st.write('UPLOAD A DOG PICTURE TO CHECK ITS EMOTION')

# Prediction tab
if selected_tab == "PREDICTION":
    st.title('PREDICTION')
    # Upload an image
    uploaded_image = st.file_uploader('Upload a dog image', type=['jpg', 'jpeg'])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert the PIL image to a NumPy array
        image_np = np.array(image)

        # Process the image using OpenCV
        processed_image = preprocess_image_cv2(image_np)

        # Make a prediction
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        predicted_class_index = np.argmax(prediction)
        predicted_class = emotion_labels[predicted_class_index]
        predicted_class_probability = prediction[0][predicted_class_index]

        # Display the prediction
        st.write(f'Predicted Emotion: {predicted_class}')
        st.write(f'Predicted Emotion Probability: {predicted_class_probability:.2f}')
