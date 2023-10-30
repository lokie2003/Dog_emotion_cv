# app.py
import os
from flask import Flask, request, render_template, Response
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_emotion(image):
    preds = model.predict(image)
    decoded_preds = decode_predictions(preds)
    # Process the decoded predictions and return the result (emotion prediction)

def camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = preprocess_image(frame)
        prediction = predict_emotion(img)

        cv2.putText(frame, f"Prediction: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
