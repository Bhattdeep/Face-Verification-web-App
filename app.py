import os
import secrets
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Import database functions
from database import (
    init_database, 
    create_user, 
    get_user_by_email, 
    get_user_by_id, 
    update_user_profile
)

# Import face verification model
from face_verification import FastFaceVerification

# Flask Application Configuration
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# File upload configuration
UPLOAD_FOLDER = 'static/profile_pictures'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize face verification
face_verifier = FastFaceVerification(threshold=0.6)

def allowed_file(filename):
    """
    Check if file extension is allowed.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_profile_picture(file, user_id):
    """
    Save profile picture with a unique name.
    """
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{user_id}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filename
    return None

# Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    User registration route with extended student details.
    """
    if request.method == 'POST':
        # Basic user details
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        profile_picture = request.files.get('profile_picture')

        # Student details
        student_details = {
            'student_name': request.form.get('student_name'),
            'father_name': request.form.get('father_name'),
            'mother_name': request.form.get('mother_name'),
            'date_of_birth': request.form.get('date_of_birth'),
            'course': request.form.get('course'),
            'semester': request.form.get('semester'),
            'branch': request.form.get('branch'),
            'section': request.form.get('section'),
            'class_roll_no': request.form.get('class_roll_no'),
            'university_roll_no': request.form.get('university_roll_no'),
            'high_school_percentage': request.form.get('high_school_percentage'),
            'intermediate_percentage': request.form.get('intermediate_percentage')
        }

        # Check if user already exists
        if get_user_by_email(email):
            flash('Email already registered')
            return redirect(url_for('register'))

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Create user
        if create_user(email, username, hashed_password, student_details):
            # If a profile picture is uploaded
            if profile_picture:
                # Get the last inserted user's ID
                user = get_user_by_email(email)
                if user:
                    filename = save_profile_picture(profile_picture, user['id'])
                    if filename:
                        # Update user with profile picture
                        update_user_profile(user['id'], username, student_details, filename)

            flash('Registration successful!')
            return redirect(url_for('login'))
        else:
            flash('Registration failed')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    User login route.
    """
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = get_user_by_email(email)
        
        if user and check_password_hash(user['password'], password):
            # Store user info in session
            session['user_id'] = user['id']
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid login credentials')
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """
    User dashboard route.
    """
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please log in to access the dashboard')
        return redirect(url_for('login'))
    
    # Retrieve user details
    user = get_user_by_id(session['user_id'])
    return render_template('dashboard.html', user=user)

@app.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    """
    Update user profile route with student details and face verification.
    """
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please log in to update your profile')
        return redirect(url_for('login'))
    
    user = get_user_by_id(session['user_id'])
    
    if request.method == 'POST':
        username = request.form['username']
        profile_picture = request.files.get('profile_picture')
        
        # Student details
        student_details = {
            'student_name': request.form.get('student_name'),
            'father_name': request.form.get('father_name'),
            'mother_name': request.form.get('mother_name'),
            'date_of_birth': request.form.get('date_of_birth'),
            'course': request.form.get('course'),
            'semester': request.form.get('semester'),
            'branch': request.form.get('branch'),
            'section': request.form.get('section'),
            'class_roll_no': request.form.get('class_roll_no'),
            'university_roll_no': request.form.get('university_roll_no'),
            'high_school_percentage': request.form.get('high_school_percentage'),
            'intermediate_percentage': request.form.get('intermediate_percentage')
        }
        
        # Perform face verification if profile picture is uploaded
        if profile_picture:
            # Get the current user's profile picture path
            current_profile_picture = user.get('profile_picture')
            
            # Save the new profile picture temporarily
            temp_filename = save_profile_picture(profile_picture, user['id'])
            
            if current_profile_picture and temp_filename:
                # Construct full file paths
                current_pic_path = os.path.join(app.config['UPLOAD_FOLDER'], current_profile_picture)
                new_pic_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                
                # Perform face verification
                verification_result = face_verifier.verify_faces(current_pic_path, new_pic_path)
                
                if verification_result['is_match']:
                    # Update profile with the new picture
                    update_user_profile(user['id'], username, student_details, temp_filename)
                    flash('Profile updated successfully!')
                    return redirect(url_for('dashboard'))
                else:
                    # Remove the temporary uploaded file
                    os.remove(new_pic_path)
                    flash('Face verification failed. Please upload a picture of the same person.')
            else:
                flash('Error processing profile picture')
        else:
            # Update profile without picture
            update_user_profile(user['id'], username, student_details)
            flash('Profile updated successfully!')
            return redirect(url_for('dashboard'))
    
    return render_template('update_profile.html', user=user)

@app.route('/logout')
def logout():
    """
    User logout route.
    """
    session.pop('user_id', None)
    flash('Logged out successfully!')
    return redirect(url_for('login'))

@app.route('/')
def index():
    """
    Index route.
    """
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Initialize database
    init_database()
    
    # Run the app
    app.run(debug=True)