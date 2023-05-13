from flask import Flask, render_template, request, redirect, url_for
import face_recognition
import cv2
import numpy as np
import sqlite3
import os



app = Flask(__name__)


# Create the database if it doesn't exist
conn = sqlite3.connect('faces.db')

# Create a table to store the faces if it doesn't exist
conn.execute('''CREATE TABLE IF NOT EXISTS faces
                 (name TEXT NOT NULL, encoding BLOB NOT NULL)''')

# Commit the changes and close the connection
conn.commit()
conn.close()

@app.route('/store', methods=['POST'])
def store():
    name = request.form['name']
    uploaded_image = request.files['image']

    # Save the uploaded image
    uploaded_image_path = './uploads/' + uploaded_image.filename
    uploaded_image.save(uploaded_image_path)

    # Load the uploaded image
    uploaded_image = face_recognition.load_image_file(uploaded_image_path)

    # Encode the face in the uploaded image
    face_encoding = face_recognition.face_encodings(uploaded_image)[0]

    # Connect to the database
    conn = sqlite3.connect('faces.db')
    cursor = conn.cursor()

    # Insert the name and face encoding into the database
    cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, face_encoding.tobytes()))

    # Commit the changes and close the database connection
    conn.commit()
    conn.close()

    # Remove the uploaded image file
    os.remove(uploaded_image_path)

    # Redirect to the index page
    return redirect('/')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Access the webcam
    video_capture = cv2.VideoCapture(0)

    # Read a single frame from the webcam
    ret, frame = video_capture.read()

    # Save the frame as an image file
    image_path = './uploads/captured_image.jpg'
    cv2.imwrite(image_path, frame)

    # Load the captured image
    captured_image = face_recognition.load_image_file(image_path)

    # Encode the faces in the captured image
    captured_encodings = face_recognition.face_encodings(captured_image)

    # Connect to the database
    conn = sqlite3.connect('faces.db')

    # Retrieve all stored face encodings and names from the database
    cursor = conn.execute('SELECT name, encoding FROM faces')
    stored_faces = cursor.fetchall()

    # Perform face verification against the stored faces in the database
    faces_match = False
    for name, encoding in stored_faces:
        stored_encoding = np.frombuffer(encoding, dtype=np.float64)
        stored_encoding = stored_encoding.reshape((128,))
        matches = face_recognition.compare_faces([stored_encoding], captured_encodings[0])
        if any(matches):
            faces_match = True
            break

    # Create an OpenCV image for display purposes
    opencv_image = cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR)

    # Draw a rectangle around the face in the captured image
    face_location = face_recognition.face_locations(captured_image)[0]
    top, right, bottom, left = face_location
    cv2.rectangle(opencv_image, (left, top), (right, bottom), (0, 255, 0), 2)

    # Save the processed image
    processed_image_path = './uploads/processed_image.jpg'
    cv2.imwrite(processed_image_path, opencv_image)

    # Release the webcam
    video_capture.release()

    if faces_match:
        return redirect(url_for('success'))
    else:
        return redirect(url_for('index'))


@app.route('/success')
def success():
    return render_template('success.html')

if __name__ == '__main__':
    app.run()
