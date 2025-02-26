import os
import cv2
import numpy as np
import face_recognition
import time

class FastFaceVerification:
    def __init__(self, threshold=0.6, max_faces=1):
        """
        Fast and efficient face verification.
        
        :param threshold: Similarity threshold for face matching
        :param max_faces: Maximum number of faces to process
        """
        self.threshold = threshold
        self.max_faces = max_faces

    def _preprocess_image(self, image_path):
        """
        Lightweight image preprocessing.
        
        :param image_path: Path to the image
        :return: Preprocessed image
        """
        # Read image
        image = cv2.imread(image_path)
        
        # Resize image to reduce processing time
        height, width = image.shape[:2]
        scale = min(1, 800 / max(height, width))
        resized_image = cv2.resize(image, None, fx=scale, fy=scale, 
                                   interpolation=cv2.INTER_AREA)
        
        return resized_image

    def detect_faces(self, image_path):
        """
        Efficient face detection.
        
        :param image_path: Path to the image
        :return: List of face locations
        """
        # Load image using face_recognition
        img = face_recognition.load_image_file(image_path)
        
        # Use HOG model for faster detection
        faces = face_recognition.face_locations(img, model='hog')
        
        # Limit to max_faces
        return faces[:self.max_faces]

    def verify_faces(self, image1_path, image2_path):
        """
        Fast face verification with improved reliability.
        
        :param image1_path: Path to the first image
        :param image2_path: Path to the second image
        :return: Verification results dictionary
        """
        start_time = time.time()
        
        try:
            # Validate file paths
            for path in [image1_path, image2_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image not found: {path}")
            
            # Detect faces
            faces1 = self.detect_faces(image1_path)
            faces2 = self.detect_faces(image2_path)
            
            # Check if faces are detected
            if not faces1 or not faces2:
                return {
                    'is_match': False,
                    'error': 'No faces detected',
                    'processing_time': time.time() - start_time
                }
            
            # Load images
            img1 = face_recognition.load_image_file(image1_path)
            img2 = face_recognition.load_image_file(image2_path)
            
            # Compute face encodings
            encodings1 = face_recognition.face_encodings(img1, faces1)
            encodings2 = face_recognition.face_encodings(img2, faces2)
            
            # Check if encoding generation was successful
            if not encodings1 or not encodings2:
                return {
                    'is_match': False,
                    'error': 'Could not generate face encodings',
                    'processing_time': time.time() - start_time
                }
            
            # Compare faces
            best_match = {
                'is_match': False,
                'face_distance': float('inf')
            }
            
            # Compare first face of each image by default
            distance = face_recognition.face_distance([encodings1[0]], encodings2[0])[0]
            
            # Determine match
            if distance <= self.threshold:
                best_match = {
                    'is_match': True,
                    'face_distance': float(distance)
                }
            
            # Add processing details
            processing_time = time.time() - start_time
            
            result = {
                'is_match': best_match['is_match'],
                'face_distance': best_match['face_distance'],
                'threshold': self.threshold,
                'processing_time': processing_time
            }
            
            return result
        
        except Exception as e:
            print(f"Face verification error: {e}")
            return {
                'is_match': False,
                'face_distance': None,
                'threshold': self.threshold,
                'error': str(e),
                'processing_time': time.time() - start_time
            }