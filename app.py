import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from streamlit_webrtc import streamlit_webrtc


from aiortc.contrib.media import MediaPlayer

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

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to 224x224 pixels
    image = image.resize((224, 224))
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image

# Function to perform image prediction
def perform_image_prediction(image):
    # Make a prediction
    prediction = model.predict(tf.expand_dims(image, axis=0))
    predicted_class_index = tf.argmax(prediction)
    predicted_class = emotion_labels[predicted_class_index]
    predicted_class_probability = prediction[0][predicted_class_index]

    return predicted_class, predicted_class_probability

# Streamlit UI
st.title('DOG EMOTION CLASSIFIER')
# Create a webrtc context
webrtc_ctx = streamlit_webrtc.VideoTransformerCanvas()

# Create tabs using selectbox
selected_tab = st.selectbox("SELECT A TAB", ["INTRODUCTION", "PREDICTION"])

# Introduction tab
if selected_tab == "INTRODUCTION":
    st.title('INTRODUCTION')
    st.write('THIS IS DOG EMOTION CLASSIFIER USING RESNET FEATURE ENGINEERING')
    st.write('UPLOAD A DOG PICTURE OR CAPTURE ONE USING YOUR CAMERA TO CHECK ITS EMOTION')

# Prediction tab
if selected_tab == "PREDICTION":
    st.title('PREDICTION')
    image_data = webrtc_ctx.frame

    if st.button("Classify Emotion"):
        if image_data is not None:
            # Convert the image to a format suitable for the model
            pil_image = Image.fromarray(image_data)
            processed_image = preprocess_image(pil_image)

            # Perform the prediction
            predicted_class, predicted_class_probability = perform_image_prediction(processed_image)

            # Display the prediction
            st.image(pil_image, caption='Captured Image', use_column_width=True)
            st.write(f'Predicted Emotion: {predicted_class}')
            st.write(f'Predicted Emotion Probability: {predicted_class_probability:.2f}')

# Release the camera when done
webrtc_ctx.stop()
