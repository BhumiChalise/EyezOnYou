from flask import Flask, render_template, request, redirect, url_for, flash, session,jsonify,send_from_directory
from flask_sqlalchemy import SQLAlchemy
import cv2
import os
import face_recognition
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import json
from functools import wraps
import logging
import mysql.connector
from sqlalchemy import text
import time
from collections import defaultdict
from flask_socketio import SocketIO, emit
import numpy as np
import uuid 
from datetime import datetime



app = Flask(__name__)
# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/eyezonyou'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')  # Folder where the images will be stored
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Use an environment variable for security
socketio = SocketIO(app)
# Initialize the database
db = SQLAlchemy(app)

def get_db_connection():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='eyezonyou'
    )
    return conn

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class RecognizedFace(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    address = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)

with app.app_context():
    db.create_all()


def save_or_update_unknown_face(face_hash):
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Check if the face already exists in the database
        query = "SELECT id, detection_count FROM unknown_faces WHERE face_hash = %s"
        cursor.execute(query, (face_hash,))
        result = cursor.fetchone()

        if result:
            # Update the detection count and last detected time
            face_id, detection_count = result
            new_count = detection_count + 1
            update_query = """
                UPDATE unknown_faces 
                SET detection_count = %s, last_detected_time = %s 
                WHERE id = %s
            """
            cursor.execute(update_query, (new_count, datetime.now(), face_id))

            # Check if detection count reaches or exceeds 5
            if new_count >= 5:
                send_danger_alert(face_hash, new_count)
        else:
            # Insert the new unknown face into the database
            insert_query = """
                INSERT INTO unknown_faces (face_hash, detection_count, last_detected_time, is_added)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (face_hash, 1, datetime.now(), 0))

        conn.commit()
    except Exception as e:
        logging.error(f"Error saving/updating unknown face: {e}")
    finally:
        cursor.close()
        conn.close()

def send_danger_alert(face_hash, count):
    """Emit a danger alert for unknown faces detected 5 or more times."""
    alert_message = f"Unknown face detected {count} times! Marked as DANGER."
    logging.warning(alert_message)
    # Emit to the dashboard
    socketio.emit('danger_alert', {
        'message': alert_message,
        'count': count,
        'face_hash': face_hash,
    })


def mark_face_as_added(face_hash):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    update_query = """
        UPDATE unknown_faces
        SET is_added = 1
        WHERE face_hash = %s
    """
    cursor.execute(update_query, (face_hash,))
    conn.commit()
    conn.close()

def get_unknown_faces():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    select_query = """
        SELECT face_hash, detection_count, last_detected_time, is_added
        FROM unknown_faces
    """
    cursor.execute(select_query)
    results = cursor.fetchall()
    conn.close()
    return results

def get_unreviewed_faces():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    select_query = """
        SELECT face_hash, detection_count, last_detected_time
        FROM unknown_faces
        WHERE is_added = 0
    """
    cursor.execute(select_query)
    results = cursor.fetchall()
    conn.close()
    return results


#for known face
def insert_recognized_face(name, age, address, phone, status):
    new_recognized_face = RecognizedFace(
        name=name,
        age=age,
        address=address,
        phone=phone,
        status=status
    )
    db.session.add(new_recognized_face)
    db.session.commit()
    print("Recognition data inserted successfully!")


## load database
def load_database():
    names = {}
    database = {}
    known_face_encodings = []

    faces = Face.query.all()
    for idx, face in enumerate(faces):
        names[face.name] = idx
        person_details = {
            'age': face.age,
            'address': face.address,
            'phone_number': face.phone_number,
            'status': face.status
        }
        database[face.name] = person_details

        if face.encodings:
            try:
                face_encodings = json.loads(face.encodings)
                known_face_encodings.append(np.array(face_encodings))
            except json.JSONDecodeError:
                logging.warning(f"Invalid encoding for {face.name}")

    return names, database, known_face_encodings

def insert_recognized_face(name, age, address, phone, status):
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        query = """
            INSERT INTO recognized_face (name, age, address, phone, status)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (name, age, address, phone, status))
        conn.commit()
        cursor.close()
        conn.close()
        print("Recognition data inserted successfully!")
    else:
        print("Failed to connect to the database")


def recognize_faces(cap):
    names, database, known_face_encodings = load_database()
    known_face_names = list(names.keys())
    recognized_names = set()
    unknown_face_counts = {}  # Dictionary to keep track of unknown face detections

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) == 0:
                continue  # Skip processing if no face distances
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.4:
                name = known_face_names[best_match_index]
                if name not in recognized_names:
                    recognized_names.add(name)
                    person_info = database.get(name)
                    if person_info:
                        recognized_face_data = {
                            'name': name,
                            'age': person_info['age'],
                            'address': person_info['address'],
                            'phone': person_info['phone_number'],
                            'status': person_info['status']
                        }
                        socketio.emit('recognized_face', recognized_face_data)
                cv2.putText(frame, name, (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                name = "Unknown"
                logging.info("Unknown face detected!")

                # Generate a unique hash for the face encoding
                face_hash = str(face_encoding.tobytes())

                # Save or update the unknown face in the MySQL database
                save_or_update_unknown_face(face_hash)

                # Emit an alert for the dashboard
                socketio.emit('unknown_face_alert', {'message': "Unknown face detected!"})

                cv2.putText(frame, name, (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Track unknown faces by their encoding (using a hashable form of face encoding)
                face_hash = str(face_encoding.tolist())  # Convert face encoding to a string for hashing
                if face_hash not in unknown_face_counts:
                    unknown_face_counts[face_hash] = 0

                unknown_face_counts[face_hash] += 1

                # Send a notification to the dashboard every time an unknown face is detected
                socketio.emit('unknown_face_alert', {'message': "Unknown face detected!"})

                # If the unknown face has been detected 5 or more times, notify the dashboard
                if unknown_face_counts[face_hash] >= 5:
                    socketio.emit('unknown_face_alert', {'message': "Unknown face detected more than 5 times!"})
                    unknown_face_counts[face_hash] = 0  # Reset the count after sending the notification

                cv2.putText(frame, name, (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)  # Red color for "Unknown"

            # Draw bounding box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display the frame with annotations
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

# Decorator for login required routes
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first!', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Define the Face model 
class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    address = db.Column(db.String(255))
    phone_number = db.Column(db.String(15))
    status = db.Column(db.String(50))
    photo = db.Column(db.String(100))
    encodings = db.Column(db.Text)  # Store encodings as JSON string



# Initialize the database and create tables if they don't exist
with app.app_context():
    db.create_all()

@app.route('/contact')
def contact():
    # Fetch data from the database
    query_contacts = "SELECT id, name, phone_number, email, photo_path FROM contact_details"
    query_socials = "SELECT contact_id, platform_name, account_link FROM social_media_accounts"

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Fetch contacts
    cursor.execute(query_contacts)
    contacts = cursor.fetchall()

    # Fetch social media accounts
    cursor.execute(query_socials)
    social_accounts = cursor.fetchall()
    conn.close()

    # Combine contact and social media data
    for contact in contacts:
        contact['social_media'] = [
            social for social in social_accounts if social['contact_id'] == contact['id']
        ]

    return render_template('contact.html', contacts=contacts)

# Route for logging in
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Hardcoded login credentials
        if username == 'admin' and password == 'password':
            session['user_id'] = 1  # Example user ID
            flash('Login successfully!', 'login_feedback')
            return redirect(url_for('index'))  # Redirecting back to the login page
        else:
            flash('Invalid username or password', 'login_feedback')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@socketio.on('unknown_face_alert')
def handle_unknown_face_alert(data):
    # When an unknown face alert is received, log it or perform any action
    print("Received unknown face alert:", data)
    # You could also update something in the app state if necessary
    # For example, use a global variable or session to hold the notification
    emit('update_notification', {'message': data['message']}, broadcast=True)

@socketio.on('start_recognition')
def handle_start_recognition():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        emit('error', {'message': 'Unable to access the camera.'})
        return

    try:
        recognized_faces = recognize_faces(cap)
        emit('recognized_faces', recognized_faces)
    except Exception as e:
        emit('error', {'message': str(e)})
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Route to enter new data
@app.route('/enter_data', methods=['GET', 'POST'])
@login_required
def enter_data():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        address = request.form['address']
        phone_number = request.form.get('phone_number', '').strip()
        status = request.form['status']
        photo_data = request.form.get('captured_photo_data')
        filename = None
        encodings = None

        # Validation for the fields
        if not name.strip() or not all(x.isalpha() or x.isspace() for x in name):
            flash("Please enter a valid name (only alphabets and spaces)!", 'enter_data_feedback')
            return redirect(url_for('enter_data'))

        if not age.isdigit() or int(age) <= 0 or int(age) >= 100:
            flash("Age is required and must be a positive number less than 100!", 'enter_data_feedback')
            return redirect(url_for('enter_data'))

        if not phone_number or not phone_number.isdigit() or len(phone_number) != 10:
            flash("Phone number is required and must be a 10-digit number!", 'enter_data_feedback')
            return redirect(url_for('enter_data'))

        if not address.strip():
            flash("Address cannot be empty!", 'enter_data_feedback')
            return redirect(url_for('enter_data'))

        if not status.strip():
            flash("Status cannot be empty!", 'enter_data_feedback')
            return redirect(url_for('enter_data'))

        if photo_data:
            try:
                img_data = base64.b64decode(photo_data.split(',')[1])
                img = Image.open(BytesIO(img_data))

                # Generate a unique filename
                unique_id = uuid.uuid4().hex[:8]
                filename = secure_filename(f"{name.replace(' ', '_')}_{unique_id}_photo.jpg")
                photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Save the image
                img.save(photo_path)

                # Process the image for face encodings
                face_image = face_recognition.load_image_file(photo_path)
                face_encodings = face_recognition.face_encodings(face_image)
                if face_encodings:
                    encodings = json.dumps(face_encodings[0].tolist())
                else:
                    flash("No face detected in the uploaded photo!", 'enter_data_feedback')
                    return redirect(url_for('enter_data'))
            except Exception as e:
                flash("Error processing the uploaded photo!", 'enter_data_feedback')
                return redirect(url_for('enter_data'))
        else:
            flash("Photo is required for face detection!", 'enter_data_feedback')
            return redirect(url_for('enter_data'))

        if not name or not age or not address or not status or not encodings or not phone_number:
           flash("All fields are required!", 'enter_data_feedback')
           return redirect(url_for('enter_data'))
        
        # Save to the database
        new_face = Face(
            name=name,
            age=int(age),
            address=address,
            phone_number=phone_number,
            status=status,
            photo=filename,
            encodings=encodings
        )
        db.session.add(new_face)
        db.session.commit()

        flash(f"Data for {name} added successfully!", 'success')
        return redirect(url_for('dashboard'))

    return render_template('enter_data.html')



# Route to view entered data
@app.route('/view_data')
@login_required
def view_data():
    faces = Face.query.all()
    return render_template('view_data.html', faces=faces)

# Route to search for a face by name
@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        search_name = request.form['name']  # Get the name from the form

        # Query the database for faces with the provided name
        matching_faces = Face.query.filter(Face.name.ilike(f"%{search_name}%")).all()

        if not matching_faces:
            flash("No matching records found!", 'danger')
            return render_template('search.html')

        if len(matching_faces) == 1:
            # If only one record matches the name, directly display the result
            return render_template('search_results.html', faces=matching_faces)

        # If multiple records with the same name are found, ask for a phone number
        phone_number = request.form.get('phone_number')

        if phone_number:
            # Filter by phone number to narrow down results
            filtered_faces = [
                face for face in matching_faces if face.phone_number == phone_number
            ]

            if not filtered_faces:
                flash("No records found with the given name and phone number!", 'danger')
                return render_template('search.html', multiple_faces=True, faces=matching_faces)

            # If a record matches both name and phone number, display the result
            return render_template('search_results.html', faces=filtered_faces)

        # Render a page to request additional information (phone number)
        flash("Multiple records found. Please provide the phone number for disambiguation.", 'info')
        return render_template('search.html', multiple_faces=True, faces=matching_faces)

    return render_template('search.html')



@app.route('/start_recognition', methods=['GET', 'POST'])
def start_recognition():
    if 'user_id' not in session:
        flash('Please log in first!', 'danger')
        return redirect(url_for('login'))

    recognized_faces = []
    unknown_face_detected = False  # Flag for detecting unknown faces
    if request.method == 'POST':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            flash('Unable to access the camera. Please check permissions or other processes.', 'danger')
            return redirect(url_for('dashboard'))

        try:
            time.sleep(1)  # Short delay for camera readiness
            recognized_faces = recognize_faces(cap)  # Start the recognition process

            # Check if an unknown face is detected
            for face in recognized_faces:
                if face == "unknown":  # You need to define how an unknown face is marked
                    unknown_face_detected = True
                    break

        except Exception as e:
            flash(f"Error during recognition: {str(e)}", 'danger')
        finally:
            cap.release()
            cv2.destroyAllWindows()

        flash('Face recognition session ended!', 'success')

        # Pass the `unknown_face_detected` flag to the template
        return render_template('start_recognition.html', recognized_faces=recognized_faces, unknown_face_detected=unknown_face_detected)

    return render_template('start_recognition.html', recognized_faces=recognized_faces, unknown_face_detected=unknown_face_detected)


@app.route('/search_by_photo', methods=['GET', 'POST'])
@login_required
def search_by_photo():
    matches_found = []  # List to store details of matched faces
    unmatched_faces = 0  # Counter for unmatched faces

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'photo' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        photo = request.files['photo']

        if photo.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if photo and allowed_file(photo.filename):
            filename = secure_filename(photo.filename)
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            photo.save(photo_path)

            try:
                # Load the uploaded image for comparison
                uploaded_image = face_recognition.load_image_file(photo_path)
                uploaded_face_encodings = face_recognition.face_encodings(uploaded_image)

                if uploaded_face_encodings:
                    # Query the database for all faces
                    faces_in_db = Face.query.all()

                    for uploaded_face_encoding in uploaded_face_encodings:
                        match_found = False

                        for face in faces_in_db:
                            # Skip entries without valid encodings
                            if not face.encodings:
                                continue

                            try:
                                db_encodings = json.loads(face.encodings)  # Deserialize encodings
                            except json.JSONDecodeError:
                                continue  # Skip invalid JSON encodings

                            # Compare the uploaded face with the database face
                            matches = face_recognition.compare_faces([db_encodings], uploaded_face_encoding)

                            if matches[0]:
                                matches_found.append(face)  # Add matched face details
                                match_found = True
                                break  # No need to check other faces in the database for this encoding

                        if not match_found:
                            unmatched_faces += 1  # Increment unmatched counter

                    # Provide feedback to the user
                    if matches_found:
                        flash(f"{len(matches_found)} matching face(s) found.", 'success')
                    if unmatched_faces > 0:
                        flash(f"{unmatched_faces} face(s) in the image did not match any record.", 'info')

                else:
                    flash('No faces detected in the uploaded image.', 'danger')

            except Exception as e:
                flash(f'An error occurred: {str(e)}', 'danger')

    return render_template(
        'search_by_photo.html', matches=matches_found, unmatched_count=unmatched_faces
    )


# Helper function to validate allowed file types
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure an 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

FRAMES_FOLDER = 'frames'
os.makedirs(FRAMES_FOLDER, exist_ok=True)


@app.route('/video_upload', methods=['GET', 'POST'])
def video_upload():
    if request.method == 'POST':
        file = request.files['video']
        person_image = request.files['person_image']
        timestamps = []

        if not file or not person_image:
            flash("Please provide both a video file and a photo of the person.", 'danger')
            return redirect(url_for('video_upload'))

        # Validate file extensions
        if not (file.filename.lower().endswith(('.mp4', '.avi', '.mov')) and person_image.filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
            flash("Invalid file format. Please upload a video file and a JPG/PNG photo.", 'danger')
            return redirect(url_for('video_upload'))

        # Ensure 'uploads' and 'frames' directories exist
        upload_folder = 'uploads'
        frames_folder = 'frames'
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(frames_folder, exist_ok=True)

        # Save video and photo
        video_path = os.path.join(upload_folder, secure_filename(file.filename))
        file.save(video_path)

        person_image_path = os.path.join(upload_folder, secure_filename(person_image.filename))
        person_image.save(person_image_path)

        # Generate person encoding
        try:
            person_image = face_recognition.load_image_file(person_image_path)
            person_encoding = face_recognition.face_encodings(person_image)[0]
        except IndexError:
            flash("The provided photo does not contain a recognizable face.", 'danger')
            return redirect(url_for('video_upload'))

        # Process the video
        video_capture = cv2.VideoCapture(video_path)
        frame_count = 0
        process_frame_interval = 10

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            if frame_count % process_frame_interval == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding in face_encodings:
                    match = face_recognition.compare_faces([person_encoding], face_encoding, tolerance=0.6)
                    if match[0]:
                        timestamp = video_capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        timestamps.append({
                            "time": f"{int(timestamp // 60)}:{int(timestamp % 60)}",
                            "image_url": f"/frames/{frame_count}.jpg"
                        })
                        # Save the frame
                        frame_path = os.path.join(frames_folder, f"{frame_count}.jpg")
                        cv2.imwrite(frame_path, frame)
                        break

            frame_count += 1

        video_capture.release()

        # Cleanup
        os.remove(video_path)
        os.remove(person_image_path)

        # Return response with rendered HTML
        return render_template('video_results.html', timestamps=timestamps)

    return render_template('video_upload.html')


@app.route('/frames/<filename>')
def serve_frame(filename):
    return send_from_directory('frames', filename)






@app.route('/delete_data', methods=['GET', 'POST'])
def delete_data():
    if 'user_id' not in session:
        flash('Please log in first!', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        name_to_delete = request.form.get('name')

        if not name_to_delete:
            flash('Name is required!', 'danger')
            return redirect(url_for('view_data'))

        # Query the database for faces matching the provided name
        faces_to_delete = Face.query.filter(Face.name.ilike(f"%{name_to_delete}%")).all()

        if not faces_to_delete:
            flash(f"No records found for name: {name_to_delete}", 'danger')
            return redirect(url_for('view_data'))

        # If multiple records are found, show them and ask for the phone number
        if len(faces_to_delete) > 1:
            return render_template('confirm_phone.html', faces=faces_to_delete)

        # If only one record, delete it directly
        face = faces_to_delete[0]
        if face.photo:
            photo_path = os.path.join(app.config['UPLOAD_FOLDER'], face.photo)
            if os.path.exists(photo_path):
                os.remove(photo_path)

        db.session.delete(face)
        db.session.commit()

        flash(f"Record for {name_to_delete} deleted successfully!", 'success')
        return redirect(url_for('dashboard'))

    return render_template('delete_data.html')

@app.route('/confirm_delete', methods=['POST'])
def confirm_delete():
    name = request.form.get('name')  # Get name from form
    phone_number = request.form.get('phone_number')  # Get phone number from form

    if not name or not phone_number:
        flash('Name and phone number are required!', 'danger')
        return redirect(url_for('delete_data'))

    # Query to find the record matching name and phone number
    face_to_delete = Face.query.filter_by(name=name, phone_number=phone_number).first()

    if not face_to_delete:
        flash('No matching record found to delete.', 'danger')
        return redirect(url_for('delete_data'))

    # Delete the record and associated photo
    if face_to_delete.photo:
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], face_to_delete.photo)
        if os.path.exists(photo_path):
            os.remove(photo_path)

    db.session.delete(face_to_delete)
    db.session.commit()

    flash(f'Record for {name} with phone number {phone_number} deleted successfully!', 'success')
    return redirect(url_for('dashboard'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/edit/<int:face_id>', methods=['GET', 'POST'])
@login_required
def edit_record(face_id):
    face = Face.query.get(face_id)
    if not face:
        flash("Record not found!", "danger")
        return redirect(url_for('search'))

    if request.method == 'POST':
        # Update face attributes with form data
        face.name = request.form['name']
        face.age = request.form['age']
        face.address = request.form['address']
        face.phone_number = request.form['phone_number']
        face.status = request.form['status']
        db.session.commit()
        flash("Record updated successfully!", "success")
        return redirect(url_for('dashboard'))

    # Render edit form with existing data
    return render_template('edit_record.html', face=face)

# Route to log out
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)