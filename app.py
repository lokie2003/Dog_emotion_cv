import streamlit as st
import numpy as np
import tensorflow as tf
import pathlib
import tensorflow_hub as hub
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

# Define a function to preprocess the image using Pillow (PIL)
def preprocess_image_pil(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
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
    st.write('UPLOAD A DOG PICTURE OR USE WEBCAM TO CHECK ITS EMOTION')

# Prediction tab
if selected_tab == "PREDICTION":
    st.title('PREDICTION')
    method = st.radio("Select method", ('Upload Image', 'Use Webcam'))

    if method == 'Upload Image':
        # Upload an image
        uploaded_image = st.file_uploader('Upload a dog image', type=['jpg', 'jpeg'])

        if uploaded_image is not None:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Process the image
            processed_image = preprocess_image_pil(image)

            # Make a prediction
            prediction = model.predict(np.expand_dims(processed_image, axis=0))
            predicted_class_index = np.argmax(prediction)
            predicted_class = emotion_labels[predicted_class_index]
            predicted_class_probability = prediction[0][predicted_class_index]

            # Display the prediction
            st.write(f'Predicted Emotion: {predicted_class}')
            st.write(f'Predicted Emotion Probability: {predicted_class_probability:.2f}')
    
    elif method == 'Use Webcam':
        st.write('Using Webcam for live prediction...')
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Resize frame to 224x224 (assuming ResNet input size)
            frame = cv2.resize(frame, (224, 224))

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make a prediction
            processed_image = frame_rgb / 255.0
            prediction = model.predict(np.expand_dims(processed_image, axis=0))
            predicted_class_index = np.argmax(prediction)
            predicted_class = emotion_labels[predicted_class_index]
            predicted_class_probability = prediction[0][predicted_class_index]

            # Display the frame with the prediction
            st.image(frame_rgb, caption=f'Predicted Emotion: {predicted_class}', use_column_width=True)
            st.write(f'Predicted Emotion Probability: {predicted_class_probability:.2f}')

        video_capture.release()
