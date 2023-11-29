import os
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, render_template
from datetime import datetime

app = Flask(__name__)

# Path to the directory containing images for attendance
image_path = 'ImagesAttendance_train'
class_names = []

# Load class names from the images in the 'ImagesAttendance' directory
for image_file in os.listdir(image_path):
    class_names.append(os.path.splitext(image_file)[0])

# Load images and extract face encodings
image_list = [face_recognition.load_image_file(f'{image_path}/{class_name}.jpeg') for class_name in class_names]
known_encodings = [face_recognition.face_encodings(image)[0] for image in image_list]

# Function to mark attendance
def mark_attendance(name):
    time_string = datetime.now().strftime('%H:%M:%S')
    with open('Attendance.txt', 'a') as file:
        file.write(f'{name}, {time_string}\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_image = request.files['image']
        if uploaded_image:
            image = face_recognition.load_image_file(uploaded_image)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                if True in matches:
                    match_index = matches.index(True)
                    detected_name = class_names[match_index]
                    mark_attendance(detected_name)
            
            return "Attendance marked for detected faces."
        else:
            return "No image uploaded."

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

