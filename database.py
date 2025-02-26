import mysql.connector
from mysql.connector import Error

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'student_management',
    'user': 'root',
    'password': 'Deep@8859#'  # Replace with your MySQL password
}

def get_db_connection():
    """Create and return a MySQL database connection."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def init_database():
    """Initialize the database and create users table."""
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(255) NOT NULL,
                password VARCHAR(255) NOT NULL,
                student_name VARCHAR(255),
                father_name VARCHAR(255),
                mother_name VARCHAR(255),
                date_of_birth DATE,
                course VARCHAR(255),
                semester VARCHAR(50),
                branch VARCHAR(255),
                section VARCHAR(50),
                class_roll_no VARCHAR(100),
                university_roll_no VARCHAR(100),
                high_school_percentage FLOAT,
                intermediate_percentage FLOAT,
                profile_picture VARCHAR(255)
            )
            ''')
            connection.commit()
    except Error as e:
        print(f"Database initialization error: {e}")
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def create_user(email, username, hashed_password, student_details):
    """Create a new user in the database."""
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor(dictionary=True)
            
            # Prepare SQL insert statement
            query = '''
            INSERT INTO users 
            (email, username, password, student_name, father_name, mother_name, 
            date_of_birth, course, semester, branch, section, class_roll_no, 
            university_roll_no, high_school_percentage, intermediate_percentage) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            
            values = (
                email, username, hashed_password,
                student_details.get('student_name'), 
                student_details.get('father_name'),
                student_details.get('mother_name'),
                student_details.get('date_of_birth'),
                student_details.get('course'),
                student_details.get('semester'),
                student_details.get('branch'),
                student_details.get('section'),
                student_details.get('class_roll_no'),
                student_details.get('university_roll_no'),
                student_details.get('high_school_percentage'),
                student_details.get('intermediate_percentage')
            )
            
            cursor.execute(query, values)
            connection.commit()
            return True
    except Error as e:
        print(f"Error creating user: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_user_by_email(email):
    """Retrieve a user by email."""
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor(dictionary=True)
            cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
            return cursor.fetchone()
    except Error as e:
        print(f"Error retrieving user: {e}")
        return None
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def get_user_by_id(user_id):
    """Retrieve a user by ID."""
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor(dictionary=True)
            cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))
            return cursor.fetchone()
    except Error as e:
        print(f"Error retrieving user: {e}")
        return None
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

def update_user_profile(user_id, username, student_details, profile_picture=None):
    """Update user profile, optionally including profile picture."""
    try:
        connection = get_db_connection()
        if connection:
            cursor = connection.cursor()
            
            # Prepare base update query
            query = '''
            UPDATE users 
            SET username = %s, student_name = %s, father_name = %s, 
            mother_name = %s, date_of_birth = %s, course = %s, 
            semester = %s, branch = %s, section = %s, 
            class_roll_no = %s, university_roll_no = %s, 
            high_school_percentage = %s, intermediate_percentage = %s
            '''
            
            # Add profile picture to update if provided
            params = [
                username, 
                student_details.get('student_name'),
                student_details.get('father_name'),
                student_details.get('mother_name'),
                student_details.get('date_of_birth'),
                student_details.get('course'),
                student_details.get('semester'),
                student_details.get('branch'),
                student_details.get('section'),
                student_details.get('class_roll_no'),
                student_details.get('university_roll_no'),
                student_details.get('high_school_percentage'),
                student_details.get('intermediate_percentage')
            ]
            
            # Add profile picture if provided
            if profile_picture:
                query += ', profile_picture = %s'
                params.append(profile_picture)
            
            # Add WHERE clause
            query += ' WHERE id = %s'
            params.append(user_id)
            
            cursor.execute(query, params)
            connection.commit()
            return True
    except Error as e:
        print(f"Error updating profile: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()