import os
import cv2
import numpy as np
import time
import insightface
from insightface.app import FaceAnalysis

class FastFaceVerification:
    def __init__(self, threshold=0.5, max_faces=1):
        """
        Fast and efficient face verification using ArcFace.
        
        :param threshold: Similarity threshold for face matching (0.0 to 1.0)
        :param max_faces: Maximum number of faces to process
        """
        self.threshold = threshold
        self.max_faces = max_faces
        
        # Initialize ArcFace model
        self.model = FaceAnalysis(name='buffalo_l')
        self.model.prepare(ctx_id=-1)  # -1 for CPU, 0 for GPU
        
        print("✓ ArcFace model initialized successfully")

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
        Efficient face detection using RetinaFace (more accurate than HOG).
        
        :param image_path: Path to the image
        :return: List of face locations in the same format as before (top, right, bottom, left)
        """
        try:
            # Load image using OpenCV
            img = cv2.imread(image_path)
            
            # Detect faces using ArcFace/RetinaFace
            faces = self.model.get(img)
            
            # Convert to same format as original (top, right, bottom, left)
            face_locations = []
            for face in faces[:self.max_faces]:
                # ArcFace returns [x1, y1, x2, y2]
                x1, y1, x2, y2 = face.bbox.astype(int)
                # Convert to (top, right, bottom, left) format
                face_location = (y1, x2, y2, x1)  # top, right, bottom, left
                face_locations.append(face_location)
            
            return face_locations
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def verify_faces(self, image1_path, image2_path):
        """
        Fast face verification with ArcFace for improved reliability.
        
        :param image1_path: Path to the first image
        :param image2_path: Path to the second image
        :return: Verification results dictionary with EXACT same structure
        """
        start_time = time.time()
        
        try:
            # Validate file paths
            for path in [image1_path, image2_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image not found: {path}")
            
            # Load images
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            
            if img1 is None or img2 is None:
                return {
                    'is_match': False,
                    'error': 'Could not read image',
                    'processing_time': time.time() - start_time
                }
            
            # Detect and get face embeddings using ArcFace
            faces1 = self.model.get(img1)
            faces2 = self.model.get(img2)
            
            # Check if faces are detected
            if not faces1 or not faces2:
                return {
                    'is_match': False,
                    'error': 'No faces detected',
                    'processing_time': time.time() - start_time
                }
            
            # Get face encodings (embeddings)
            # ArcFace model returns embeddings in face.normed_embedding
            encoding1 = faces1[0].normed_embedding if len(faces1) > 0 else None
            encoding2 = faces2[0].normed_embedding if len(faces2) > 0 else None
            
            # Check if encoding generation was successful
            if encoding1 is None or encoding2 is None:
                return {
                    'is_match': False,
                    'error': 'Could not generate face encodings',
                    'processing_time': time.time() - start_time
                }
            
            # Calculate cosine similarity
            similarity = np.dot(encoding1, encoding2) / (
                np.linalg.norm(encoding1) * np.linalg.norm(encoding2)
            )
            
            # Convert similarity to distance (1 - similarity)
            # This maintains the same logic: lower distance = more similar
            distance = 1.0 - similarity
            
            # Determine match (using threshold on distance)
            is_match = distance <= self.threshold
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Return EXACT same structure as original
            result = {
                'is_match': bool(is_match),
                'face_distance': float(distance),
                'threshold': float(self.threshold),
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