from flask import Flask, request, render_template, Response
import numpy as np
import tensorflow as tf
import pathlib
import tensorflow_hub as hub
from PIL import Image
import cv2

app = Flask(__name)

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

def gen_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        success, frame = video_capture.read()
        if not success:
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
        # Display the prediction on the frame
        cv2.putText(frame, f'Predicted Emotion: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Probability: {predicted_class_probability:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
