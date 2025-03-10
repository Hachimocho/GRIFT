import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
import dlib
from deepface import DeepFace
from fer import FER
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import gc
import logging
import time
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse
import sys

# Global debug flag
DEBUG_MODE = False

def enable_debug_logging():
    """Enable debug logging to file."""
    global DEBUG_MODE
    DEBUG_MODE = True
    
    # Configure logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file handler for debug logs
    log_file = os.path.join(log_dir, 'symmetry_debug.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # Configure logger
    logger = logging.getLogger('symmetry_debug')
    logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.WARNING)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger with warning level by default
logger = logging.getLogger('symmetry_debug')
logger.setLevel(logging.WARNING)

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Set up file handler
log_file = os.path.join(log_dir, 'symmetry_debug.log')
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))

# Set up console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(levelname)s: %(message)s'
))

# Configure logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Set up logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

# Check for CUDA availability and set memory management
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Set to use less GPU memory
    #torch.cuda.set_per_process_memory_fraction(0.3)  # Use only 30% of GPU memory
    #logger.info(f"Using device: {DEVICE} with limited memory")
else:
    logger.info(f"Using device: {DEVICE}")

# Initialize global models to avoid reloading
logger.info("Initializing face detector...")
face_detector = dlib.get_frontal_face_detector()
try:
    if torch.cuda.is_available():
        dlib.DLIB_USE_CUDA = True
        logger.info("DLIB using CUDA")
    landmark_predictor = dlib.shape_predictor('./utils/shape_predictor_68_face_landmarks.dat')
except Exception as e:
    logger.error(f"Error loading shape predictor: {str(e)}")
    landmark_predictor = None

# Initialize FER (CPU only)
logger.info("Initializing emotion detector...")
emotion_detector = FER(mtcnn=True)

# Initialize FaceNet models (keep only resnet on GPU, MTCNN on CPU)
logger.info("Initializing face recognition models...")
mtcnn = MTCNN(device='cpu')  # MTCNN on CPU to save GPU memory
resnet = InceptionResnetV1(pretrained='vggface2').to(DEVICE)
resnet.eval()

# Configuration for attribute calculation
ATTRIBUTE_CONFIG = {
    'face_embedding': {
        'enabled': True,
        'description': 'Calculate face embeddings using FaceNet'
    },
    'quality_metrics': {
        'enabled': True,
        'description': 'Calculate image quality metrics (blur, brightness, contrast, compression)',
        'components': {
            'blur': True,
            'brightness_contrast': True,
            'compression': True
        }
    },
    'alignment': {
        'enabled': True,
        'description': 'Calculate face alignment angles (yaw, pitch, roll)'
    },
    'symmetry': {
        'enabled': True,
        'description': 'Calculate facial symmetry metrics',
        'components': {
            'eyes': True,
            'mouth': True,
            'nose': True
        }
    },
    'emotions': {
        'enabled': True,
        'description': 'Calculate emotion scores using FER'
    },
    'deepface': {
        'enabled': False,
        'description': 'Calculate age, gender, and race using DeepFace',
        'components': {
            'age': True,
            'gender': True,
            'race': True
        }
    }
}

def get_default_attributes() -> Dict[str, Any]:
    """Get default attributes based on enabled configuration."""
    defaults = {}
    
    if ATTRIBUTE_CONFIG['face_embedding']['enabled']:
        defaults['face_embedding'] = np.zeros(512)
        
    if ATTRIBUTE_CONFIG['quality_metrics']['enabled']:
        defaults['quality_metrics'] = {
            'blur_score': -1.0,
            'brightness': -1.0,
            'contrast': -1.0,
            'compression_score': -1.0
        }
        
    # if ATTRIBUTE_CONFIG['alignment']['enabled']:
    #     defaults['alignment'] = {
    #         'face_yaw': -1.0,
    #         'face_pitch': -1.0,
    #         'face_roll': -1.0
    #     }
        
    if ATTRIBUTE_CONFIG['symmetry']['enabled']:
        defaults['symmetry'] = {
            'eye_ratio': -1.0,
            'mouth_ratio': -1.0,
            'nose_ratio': -1.0,
            'overall_symmetry': -1.0
        }
        
    if ATTRIBUTE_CONFIG['emotions']['enabled']:
        defaults['emotion_scores'] = {
            'angry': -1.0,
            'disgust': -1.0,
            'fear': -1.0,
            'happy': -1.0,
            'sad': -1.0,
            'surprise': -1.0,
            'neutral': -1.0
        }
        
    if ATTRIBUTE_CONFIG['deepface']['enabled']:
        defaults.update({
            'age': -1,
            'gender': 'unknown',
            'race': 'unknown',
            'race_scores': {
                'asian': -1.0,
                'indian': -1.0,
                'black': -1.0,
                'white': -1.0,
                'middle eastern': -1.0,
                'latino hispanic': -1.0
            }
        })
    
    defaults['error'] = None
    return defaults

def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate image blur score using Laplacian variance.
    Lower values indicate more blur."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_brightness_contrast(image: np.ndarray) -> Tuple[float, float]:
    """Calculate average brightness and contrast of image."""
    if len(image.shape) == 3:
        # Convert to LAB color space and use L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
    else:
        l_channel = image
        
    brightness = np.mean(l_channel)
    contrast = np.std(l_channel)
    return brightness, contrast

def calculate_compression_score(image: np.ndarray) -> float:
    """Estimate JPEG compression level using DCT coefficients."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    h = h - h % 8
    w = w - w % 8
    gray = gray[:h, :w]
    
    # Calculate DCT coefficients
    dct = cv2.dct(np.float32(gray))
    # Calculate the number of non-zero coefficients
    compression_score = np.count_nonzero(dct) / (h * w)
    return compression_score

# def calculate_face_alignment(image: np.ndarray) -> Tuple[float, float, float]:
#     """Calculate face alignment angles using facial landmarks.
#     Returns (yaw, pitch, roll) angles in degrees."""

#     # STEP 2: Create an FaceLandmarker object.
#     base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
#     options = vision.FaceLandmarkerOptions(base_options=base_options,
#                                         output_face_blendshapes=True,
#                                         output_facial_transformation_matrixes=True,
#                                         num_faces=1)
#     detector = vision.FaceLandmarker.create_from_options(options)

#     # STEP 3: Load the input image.
#     image = mp.Image.create_from_file("image.png")

#     # STEP 4: Detect face landmarks from the input image.
#     detection_result = detector.detect(image)

#     # STEP 5: Process the detection result. In this case, visualize it.
#     annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
#     cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    

    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # faces = face_cascade.detectMultiScale(image, 1.1, 4)
    
    # if len(faces) == 0:
    #     return 0.0, 0.0, 0.0
        
    # # Get the largest face
    # x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
    # face = image[y:y+h, x:x+w]
    
    # # Calculate rough angles based on face bounding box aspect ratio
    # aspect_ratio = w / h
    # yaw = (aspect_ratio - 1.0) * 45  # Rough estimate
    # pitch = 0.0  # Would need more sophisticated detection for pitch
    # roll = 0.0  # Would need more sophisticated detection for roll
    
    # return yaw, pitch, roll

def calculate_region_symmetry(left_points: np.ndarray, right_points: np.ndarray, midline: np.ndarray) -> float:
    """Calculate symmetry score for a pair of facial regions.
    
    Returns:
        float: Symmetry score where:
            1.0 = perfect symmetry (distances match exactly)
            0.0 = neutral symmetry (differences equal mean distance)
            negative = asymmetric (differences larger than mean distance)
    """
    try:
        # Ensure we have points to work with
        if len(left_points) == 0 or len(right_points) == 0:
            logger.warning("Skipping symmetry calculation: No points provided")
            return 0.0
            
        # Convert points to numpy arrays if they aren't already
        left = np.array(left_points)
        right = np.array(right_points)
        mid = np.array(midline)
        
        if DEBUG_MODE:
            logger.debug(f"Left points: {left.tolist()}")
            logger.debug(f"Right points: {right.tolist()}")
            logger.debug(f"Midline: {mid.tolist()}")
        
        # If points don't match in number, interpolate to match
        if len(left) != len(right):
            if DEBUG_MODE:
                logger.debug(f"Interpolating points from Left: {len(left)}, Right: {len(right)}")
            # Use the smaller number of points
            n_points = min(len(left), len(right))
            
            # Interpolate both sides to have the same number of points
            left_x = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(left)), left[:, 0])
            left_y = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(left)), left[:, 1])
            right_x = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(right)), right[:, 0])
            right_y = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(right)), right[:, 1])
            
            left = np.column_stack((left_x, left_y))
            right = np.column_stack((right_x, right_y))
            
            if DEBUG_MODE:
                logger.debug(f"After interpolation - Left: {left.tolist()}, Right: {right.tolist()}")
        
        # Calculate distances from each point to the midline
        # Get midline direction and normalize it
        midline_vec = mid[1] - mid[0]
        midline_length = np.linalg.norm(midline_vec)
        
        if midline_length == 0:
            logger.warning("Midline length is zero")
            return 0.0
            
        # Normalize midline vector
        midline_unit = midline_vec / midline_length
        
        # Calculate perpendicular distances using dot product with perpendicular vector
        # Perpendicular vector is (-y, x) for vector (x, y)
        perp_vec = np.array([-midline_unit[1], midline_unit[0]])
        
        # Calculate distances using dot product with perpendicular vector
        left_distances = np.abs(np.dot(left - mid[0], perp_vec))
        right_distances = np.abs(np.dot(right - mid[0], perp_vec))
        
        if DEBUG_MODE:
            logger.debug(f"Left distances: {left_distances.tolist()}")
            logger.debug(f"Right distances: {right_distances.tolist()}")
        
        # Compare the distances
        distance_diff = np.abs(left_distances - right_distances)
        mean_distance = np.mean(np.concatenate([left_distances, right_distances]))
        
        if DEBUG_MODE:
            logger.debug(f"Distance differences: {distance_diff.tolist()}")
            logger.debug(f"Mean distance: {mean_distance}")
        
        # Avoid division by zero
        if mean_distance == 0:
            logger.warning("Mean distance is zero, returning 0 symmetry score")
            return 0.0
            
        # Calculate symmetry score without clamping
        symmetry_score = 1.0 - np.mean(distance_diff) / mean_distance
        
        if DEBUG_MODE:
            logger.debug(f"Raw symmetry score: {symmetry_score}")
        
        return symmetry_score
        
    except Exception as e:
        logger.warning(f"Error calculating region symmetry: {str(e)}")
        return 0.0

def draw_landmarks(image: np.ndarray, landmarks: Dict[str, np.ndarray], nose_bridge: np.ndarray) -> None:
    """Draw facial landmarks on the image with different colors for each feature."""
    colors = {
        'left_eye': (0, 255, 0),     # Green
        'right_eye': (255, 128, 0),  # Orange
        'left_mouth': (255, 0, 0),   # Red
        'right_mouth': (128, 0, 255),# Purple
        'left_nose': (0, 255, 255),  # Yellow
        'right_nose': (255, 255, 0),  # Cyan
        'nose_bridge': (255, 255, 255)# White
    }
    
    # Draw points and lines
    for feature, points in landmarks.items():
        color = colors[feature]
        
        # Draw points
        for point in points:
            cv2.circle(image, tuple(point.astype(int)), 3, color, -1)
            
        # Draw lines connecting points
        if len(points) > 1:
            for i in range(len(points) - 1):
                pt1 = tuple(points[i].astype(int))
                pt2 = tuple(points[i + 1].astype(int))
                cv2.line(image, pt1, pt2, color, 2)
    
    # Draw nose bridge line
    if nose_bridge is not None:
        cv2.line(image, 
                tuple(nose_bridge[0].astype(int)), 
                tuple(nose_bridge[1].astype(int)), 
                colors['nose_bridge'], 2)

def calculate_facial_symmetry(image: np.ndarray) -> Dict[str, float]:
    """Calculate facial symmetry using facial landmarks."""
    try:
        if landmark_predictor is None:
            return {
                'eye_ratio': 0.0,
                'mouth_ratio': 0.0,
                'nose_ratio': 0.0,
                'overall_symmetry': 0.0
            }
        
        # Detect faces
        faces = face_detector(image)
        if len(faces) == 0:
            return {
                'eye_ratio': 0.0,
                'mouth_ratio': 0.0,
                'nose_ratio': 0.0,
                'overall_symmetry': 0.0
            }
        
        # Get landmarks
        landmarks = landmark_predictor(image, faces[0])
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Define the nose bridge (midline)
        nose_bridge = np.array([points[27], points[30]])  # Points from nose bridge
        
        # Define regions carefully
        left_eye = points[36:42]    # Left eye
        right_eye = points[42:48]   # Right eye
        left_mouth = points[48:52]  # Left mouth (corner to middle)
        right_mouth = points[52:55] # Right mouth (middle to corner)
        left_nose = points[31:34]   # Left nose (31-33: left nostril)
        right_nose = points[34:36]  # Right nose (34-35: right nostril)
        
        # Store landmarks for visualization
        landmarks_dict = {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'left_mouth': left_mouth,
            'right_mouth': right_mouth,
            'left_nose': left_nose,
            'right_nose': right_nose,
            'nose_bridge': nose_bridge
        }
        
        # Calculate symmetry scores
        if DEBUG_MODE:
            logger.debug("Calculating eye symmetry...")
        eye_sym = calculate_region_symmetry(left_eye, np.flip(right_eye, axis=0), nose_bridge)
        
        if DEBUG_MODE:
            logger.debug("Calculating mouth symmetry...")
        mouth_sym = calculate_region_symmetry(left_mouth, np.flip(right_mouth, axis=0), nose_bridge)
        
        if DEBUG_MODE:
            logger.debug("Calculating nose symmetry...")
        nose_sym = calculate_region_symmetry(left_nose, np.flip(right_nose, axis=0), nose_bridge)
        
        # Calculate overall symmetry
        overall_sym = np.mean([eye_sym, mouth_sym, nose_sym])
        
        if DEBUG_MODE:
            logger.debug(f"Symmetry scores - Eye: {eye_sym:.3f}, Mouth: {mouth_sym:.3f}, Nose: {nose_sym:.3f}, Overall: {overall_sym:.3f}")
        
        return {
            'eye_ratio': float(eye_sym),
            'mouth_ratio': float(mouth_sym),
            'nose_ratio': float(nose_sym),
            'overall_symmetry': float(overall_sym),
            '_debug': {
                'landmarks': landmarks_dict,
                'nose_bridge': nose_bridge
            } if DEBUG_MODE else None
        }
        
    except Exception as e:
        logger.warning(f"Error in facial symmetry calculation: {str(e)}")
        return {
            'eye_ratio': 0.0,
            'mouth_ratio': 0.0,
            'nose_ratio': 0.0,
            'overall_symmetry': 0.0,
            '_debug': None
        }

def get_emotions(face_img: np.ndarray) -> Dict[str, float]:
    """Get emotion predictions with GPU-accelerated preprocessing."""
    try:
        # FER doesn't support direct GPU usage, but we can optimize the preprocessing
        emotions = emotion_detector.detect_emotions(face_img)
        if emotions:
            return emotions[0]['emotions']
        return {}
    except Exception as e:
        logger.error(f"Error detecting emotions: {str(e)}")
        return {}

def detect_facial_attributes(image: np.ndarray) -> Dict[str, Any]:
    """Detect facial attributes using DeepFace."""
    try:
        analysis = DeepFace.analyze(
            image,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False
        )
        
        return {
            'hair_color': analysis.get('dominant_hair_color', 'unknown'),
            'eye_color': analysis.get('dominant_eye_color', 'unknown'),
            'has_beard': analysis.get('has_beard', False),
            'has_mustache': analysis.get('has_mustache', False),
            'wearing_glasses': analysis.get('wearing_glasses', False),
            'wearing_sunglasses': analysis.get('wearing_sunglasses', False)
        }
    except:
        return {
            'hair_color': 'unknown',
            'eye_color': 'unknown',
            'has_beard': False,
            'has_mustache': False,
            'wearing_glasses': False,
            'wearing_sunglasses': False
        }

def generate_face_embedding(image: np.ndarray) -> np.ndarray:
    """Generate face embeddings using FaceNet."""
    try:
        # Convert to PIL Image
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect and align face
        face = mtcnn(img)
        if face is None:
            return np.zeros(512)
            
        # Generate embedding
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0).to(DEVICE))
        
        return embedding.cpu().numpy().flatten()
    except:
        return np.zeros(512)

def compute_embedding_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two face embeddings."""
    if np.all(embedding1 == 0) or np.all(embedding2 == 0):
        return 0.0
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def check_memory_availability():
    """Check if we have enough GPU memory available."""
    if torch.cuda.is_available():
        try:
            # Try allocating a small tensor
            torch.cuda.empty_cache()
            test_tensor = torch.zeros((1, 3, 160, 160), device=DEVICE)
            del test_tensor
            torch.cuda.empty_cache()
            return True
        except RuntimeError:
            logger.error("Not enough GPU memory available")
            return False
    return True

def process_cpu_tasks(img, img_path, config):
    """Process CPU-bound tasks for a single image."""
    result = {}
    
    try:
        # Quality metrics
        if config['quality_metrics']['enabled']:
            metrics = {}
            if config['quality_metrics']['components']['blur']:
                metrics['blur_score'] = calculate_blur_score(img)
            if config['quality_metrics']['components']['brightness_contrast']:
                brightness, contrast = calculate_brightness_contrast(img)
                metrics['brightness'] = brightness
                metrics['contrast'] = contrast
            if config['quality_metrics']['components']['compression']:
                metrics['compression_score'] = calculate_compression_score(img)
            result['quality_metrics'] = metrics

        # Symmetry
        if config['symmetry']['enabled']:
            symmetry_results = calculate_facial_symmetry(img)
            if symmetry_results:
                result['symmetry'] = {k: v for k, v in symmetry_results.items() if not k.startswith('_')}
                result['_debug'] = symmetry_results['_debug']

    except Exception as e:
        logger.error(f"Error in CPU tasks for {img_path}: {str(e)}")
        result['error'] = str(e)

    return result

def process_image_batch(image_paths: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
    """Process a batch of images in parallel using GPU acceleration and multiprocessing for CPU tasks."""
    results = []
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    # Initialize emotion detector once
    if ATTRIBUTE_CONFIG['emotions']['enabled']:
        emotion_detector = FER(mtcnn=False)
    
    # Create process pool for CPU tasks
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU for system
    pool = Pool(processes=num_processes)
    
    try:
        for batch_idx in range(total_batches):
            if batch_idx % 10 == 0:
                # Clear GPU memory periodically
                clear_gpu_memory()
                
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            # Pre-load all images in batch
            batch_images = []
            batch_faces = []
            batch_chips = []
            valid_indices = []
            
            for idx, img_path in enumerate(batch_paths):
                try:
                    # Validate and load image
                    is_valid, error_msg = validate_image(img_path)
                    if not is_valid:
                        logger.error(f"Invalid image {img_path}: {error_msg}")
                        results.append({**get_default_attributes(), 'image_path': img_path, 'image_id': img_path.split('/')[-1], 'error': f"Invalid image: {error_msg}"})
                        continue
                        
                    img = cv2.imread(img_path)
                    if img is None:
                        results.append({**get_default_attributes(), 'image_path': img_path, 'image_id': img_path.split('/')[-1], 'error': "Failed to load image"})
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Detect face
                    faces = face_detector(img)
                    if not faces:
                        results.append({**get_default_attributes(), 'image_path': img_path, 'image_id': img_path.split('/')[-1], 'error': "No face detected"})
                        continue
                        
                    # Get face chip
                    shape = landmark_predictor(img, faces[0])
                    face_chip = dlib.get_face_chip(img, shape)
                    
                    # Store valid image data
                    batch_images.append(img)
                    batch_faces.append(faces[0])
                    batch_chips.append(face_chip)
                    valid_indices.append(idx)
                    
                except Exception as e:
                    logger.error(f"Error preprocessing {img_path}: {str(e)}")
                    results.append({**get_default_attributes(), 'image_path': img_path, 'image_id': img_path.split('/')[-1], 'error': f"Preprocessing error: {str(e)}"})
            
            if not valid_indices:
                continue
            
            # Process valid images in batch
            batch_results = [get_default_attributes() for _ in range(len(valid_indices))]
            
            # Face embeddings (batch processing on GPU)
            if ATTRIBUTE_CONFIG['face_embedding']['enabled']:
                try:
                    # Process all face chips in one batch
                    face_tensors = torch.stack([
                        torch.from_numpy(cv2.resize(chip, (160, 160))).float().permute(2, 0, 1)
                        for chip in batch_chips
                    ]).to(DEVICE)
                    
                    with torch.no_grad():
                        embeddings = resnet(face_tensors).cpu().numpy()
                    
                    for idx, embedding in enumerate(embeddings):
                        batch_results[idx]['face_embedding'] = embedding
                        
                except Exception as e:
                    logger.warning(f"Batch face embedding failed: {str(e)}")
            
            # Process CPU-bound tasks in parallel
            cpu_task_args = [(img, batch_paths[orig_idx], ATTRIBUTE_CONFIG) 
                            for img, orig_idx in zip(batch_images, valid_indices)]
            
            cpu_results = pool.starmap(process_cpu_tasks, cpu_task_args)
            
            # Process emotions in batch (if enabled)
            if ATTRIBUTE_CONFIG['emotions']['enabled']:
                try:
                    emotions_batch = emotion_detector.detect_emotions(batch_images)
                    for idx, emotions in enumerate(emotions_batch):
                        if emotions and len(emotions) > 0:
                            batch_results[idx]['emotion_scores'] = emotions[0]['emotions']
                except Exception as e:
                    logger.warning(f"Batch emotion detection failed: {str(e)}")
            
            # Merge CPU results with batch results
            for idx, (cpu_result, orig_idx) in enumerate(zip(cpu_results, valid_indices)):
                batch_results[idx].update(cpu_result)
                img_path = batch_paths[orig_idx]
                batch_results[idx]['image_path'] = img_path
                batch_results[idx]['image_id'] = img_path.split('/')[-1]
            
            results.extend(batch_results)
            
            # Explicitly clear some memory
            del batch_images
            del batch_faces
            del batch_chips
            if 'face_tensors' in locals():
                del face_tensors
            gc.collect()
            
    finally:
        # Clean up
        pool.close()
        pool.join()
        clear_gpu_memory()
    
    return results

def validate_image(image_path: str) -> Tuple[bool, str]:
    """Validate if an image can be properly loaded and processed.
    
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        # Try reading with PIL first
        with Image.open(image_path) as img:
            # Check if image can be loaded
            img.verify()
            
        # Try reading with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return False, "OpenCV failed to load image"
            
        # Check image dimensions
        if img.shape[0] < 10 or img.shape[1] < 10:
            return False, f"Image too small: {img.shape}"
            
        # Check if image is grayscale
        if len(img.shape) < 3:
            return False, "Image is grayscale"
            
        return True, ""
        
    except Exception as e:
        return False, str(e)

def process_single_image(image_path: str) -> Dict[str, Any]:
    """Process a single image to extract facial attributes."""
    # Get default attributes based on config
    results = get_default_attributes()
    results['image_path'] = image_path
    results['image_id'] = image_path
    
    try:
        # Validate image first
        is_valid, error_msg = validate_image(image_path)
        if not is_valid:
            logger.error(f"Invalid image {image_path}: {error_msg}")
            results['error'] = f"Invalid image: {error_msg}"
            return results
            
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            results['error'] = "Failed to load image"
            return results
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get face location
        face_locations = face_detector(img)
        if not face_locations:
            logger.warning(f"No face detected in {image_path}")
            results['error'] = "No face detected"
            return results
            
        # Use the first face found
        face_location = face_locations[0]
        
        try:
            # Extract face landmarks
            shape = landmark_predictor(img, face_location)
            face_chip = dlib.get_face_chip(img, shape)
            
            # Face embedding
            if ATTRIBUTE_CONFIG['face_embedding']['enabled']:
                try:
                    face_tensor = torch.from_numpy(cv2.resize(face_chip, (160, 160))).float()
                    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        embedding = resnet(face_tensor).cpu().numpy().flatten()
                        results['face_embedding'] = embedding
                except Exception as e:
                    logger.warning(f"Face embedding failed for {image_path}: {str(e)}")
            
            # Quality metrics
            if ATTRIBUTE_CONFIG['quality_metrics']['enabled']:
                try:
                    metrics = {}
                    if ATTRIBUTE_CONFIG['quality_metrics']['components']['blur']:
                        metrics['blur_score'] = calculate_blur_score(img)
                    if ATTRIBUTE_CONFIG['quality_metrics']['components']['brightness_contrast']:
                        brightness, contrast = calculate_brightness_contrast(img)
                        metrics['brightness'] = brightness
                        metrics['contrast'] = contrast
                    if ATTRIBUTE_CONFIG['quality_metrics']['components']['compression']:
                        metrics['compression_score'] = calculate_compression_score(img)
                    results['quality_metrics'].update(metrics)
                except Exception as e:
                    logger.warning(f"Quality metrics calculation failed for {image_path}: {str(e)}")
            
            # Alignment
            # if ATTRIBUTE_CONFIG['alignment']['enabled']:
            #     try:
            #         yaw, pitch, roll = calculate_face_alignment(img)
            #         results['alignment'].update({
            #             'face_yaw': yaw,
            #             'face_pitch': pitch,
            #             'face_roll': roll
            #         })
            #     except Exception as e:
            #         logger.warning(f"Alignment calculation failed for {image_path}: {str(e)}")
            
            # Symmetry
            if ATTRIBUTE_CONFIG['symmetry']['enabled']:
                try:
                    symmetry_results = calculate_facial_symmetry(img)
                    if symmetry_results:
                        results['symmetry'].update({k: v for k, v in symmetry_results.items() if not k.startswith('_')})
                        results['_debug'] = symmetry_results['_debug']
                except Exception as e:
                    logger.warning(f"Symmetry calculation failed for {image_path}: {str(e)}")
            
            # Emotions
            if ATTRIBUTE_CONFIG['emotions']['enabled']:
                try:
                    emotion_detector = FER(mtcnn=False)
                    emotions = emotion_detector.detect_emotions(img)
                    if emotions and len(emotions) > 0:
                        results['emotion_scores'] = emotions[0]['emotions']
                except Exception as e:
                    logger.warning(f"Emotion detection failed for {image_path}: {str(e)}")
            
            # DeepFace attributes
            required_deepface = (
                ATTRIBUTE_CONFIG['deepface']['enabled'] and (
                    ATTRIBUTE_CONFIG['deepface']['components']['age'] or
                    ATTRIBUTE_CONFIG['deepface']['components']['gender'] or
                    ATTRIBUTE_CONFIG['deepface']['components']['race']
                )
            )
            if required_deepface:
                try:
                    attributes = DeepFace.analyze(
                        img, 
                        actions=['age', 'gender', 'race'] if all(ATTRIBUTE_CONFIG['deepface']['components'].values())
                               else [k for k, v in ATTRIBUTE_CONFIG['deepface']['components'].items() if v],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    
                    if isinstance(attributes, list):
                        attributes = attributes[0]
                    
                    if ATTRIBUTE_CONFIG['deepface']['components']['age']:
                        results['age'] = attributes.get('age', -1)
                    if ATTRIBUTE_CONFIG['deepface']['components']['gender']:
                        results['gender'] = attributes.get('gender', 'unknown')
                    if ATTRIBUTE_CONFIG['deepface']['components']['race']:
                        results['race'] = attributes.get('dominant_race', 'unknown')
                        results['race_scores'] = attributes.get('race', results['race_scores'])
                        
                except Exception as e:
                    logger.warning(f"DeepFace analysis failed for {image_path}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error processing face in {image_path}: {str(e)}")
            results['error'] = f"Face processing error: {str(e)}"
            
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        results['error'] = f"Processing error: {str(e)}"
    
    return results

def process_image_batch(image_paths: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
    """Process a batch of images in parallel using GPU acceleration."""
    results = []
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        # Pre-load all images in batch
        batch_images = []
        batch_faces = []
        batch_chips = []
        valid_indices = []
        
        for idx, img_path in enumerate(batch_paths):
            try:
                # Validate and load image
                is_valid, error_msg = validate_image(img_path)
                if not is_valid:
                    logger.error(f"Invalid image {img_path}: {error_msg}")
                    results.append({**get_default_attributes(), 'image_path': img_path, 'image_id': img_path.split('/')[-1], 'error': f"Invalid image: {error_msg}"})
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    results.append({**get_default_attributes(), 'image_path': img_path, 'image_id': img_path.split('/')[-1], 'error': "Failed to load image"})
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detect face
                faces = face_detector(img)
                if not faces:
                    results.append({**get_default_attributes(), 'image_path': img_path, 'image_id': img_path.split('/')[-1], 'error': "No face detected"})
                    continue
                    
                # Get face chip
                shape = landmark_predictor(img, faces[0])
                face_chip = dlib.get_face_chip(img, shape)
                
                # Store valid image data
                batch_images.append(img)
                batch_faces.append(faces[0])
                batch_chips.append(face_chip)
                valid_indices.append(idx)
                
            except Exception as e:
                logger.error(f"Error preprocessing {img_path}: {str(e)}")
                results.append({**get_default_attributes(), 'image_path': img_path, 'image_id': img_path.split('/')[-1], 'error': f"Preprocessing error: {str(e)}"})
        
        if not valid_indices:
            continue
        
        # Process valid images in batch
        batch_results = [get_default_attributes() for _ in range(len(valid_indices))]
        
        # Face embeddings (batch processing)
        if ATTRIBUTE_CONFIG['face_embedding']['enabled']:
            try:
                # Process all face chips in one batch
                face_tensors = torch.stack([
                    torch.from_numpy(cv2.resize(chip, (160, 160))).float().permute(2, 0, 1)
                    for chip in batch_chips
                ]).to(DEVICE)
                
                with torch.no_grad():
                    embeddings = resnet(face_tensors).cpu().numpy()
                
                for idx, embedding in enumerate(embeddings):
                    batch_results[idx]['face_embedding'] = embedding
                    
            except Exception as e:
                logger.warning(f"Batch face embedding failed: {str(e)}")
        
        # Process other attributes for each image
        for batch_idx, (img, face, chip, orig_idx) in enumerate(zip(batch_images, batch_faces, batch_chips, valid_indices)):
            result = batch_results[batch_idx]
            img_path = batch_paths[orig_idx]
            result['image_path'] = img_path
            result['image_id'] = img_path
            
            try:
                # Quality metrics
                if ATTRIBUTE_CONFIG['quality_metrics']['enabled']:
                    try:
                        metrics = {}
                        if ATTRIBUTE_CONFIG['quality_metrics']['components']['blur']:
                            metrics['blur_score'] = calculate_blur_score(img)
                        if ATTRIBUTE_CONFIG['quality_metrics']['components']['brightness_contrast']:
                            brightness, contrast = calculate_brightness_contrast(img)
                            metrics['brightness'] = brightness
                            metrics['contrast'] = contrast
                        if ATTRIBUTE_CONFIG['quality_metrics']['components']['compression']:
                            metrics['compression_score'] = calculate_compression_score(img)
                        result['quality_metrics'].update(metrics)
                    except Exception as e:
                        logger.warning(f"Quality metrics calculation failed for {img_path}: {str(e)}")
                
                # Alignment
                # if ATTRIBUTE_CONFIG['alignment']['enabled']:
                #     try:
                #         yaw, pitch, roll = calculate_face_alignment(img)
                #         result['alignment'].update({
                #             'face_yaw': yaw,
                #             'face_pitch': pitch,
                #             'face_roll': roll
                #         })
                #     except Exception as e:
                #         logger.warning(f"Alignment calculation failed for {img_path}: {str(e)}")
                
                # Symmetry
                if ATTRIBUTE_CONFIG['symmetry']['enabled']:
                    try:
                        symmetry_results = calculate_facial_symmetry(img)
                        if symmetry_results:
                            result['symmetry'].update({k: v for k, v in symmetry_results.items() if not k.startswith('_')})
                            result['_debug'] = symmetry_results['_debug']
                    except Exception as e:
                        logger.warning(f"Symmetry calculation failed for {img_path}: {str(e)}")
                
                # Emotions
                if ATTRIBUTE_CONFIG['emotions']['enabled']:
                    try:
                        emotion_detector = FER(mtcnn=False)
                        emotions = emotion_detector.detect_emotions(img)
                        if emotions and len(emotions) > 0:
                            result['emotion_scores'] = emotions[0]['emotions']
                    except Exception as e:
                        logger.warning(f"Emotion detection failed for {img_path}: {str(e)}")
                
                # DeepFace attributes
                required_deepface = (
                    ATTRIBUTE_CONFIG['deepface']['enabled'] and (
                        ATTRIBUTE_CONFIG['deepface']['components']['age'] or
                        ATTRIBUTE_CONFIG['deepface']['components']['gender'] or
                        ATTRIBUTE_CONFIG['deepface']['components']['race']
                    )
                )
                if required_deepface:
                    try:
                        attributes = DeepFace.analyze(
                            img, 
                            actions=['age', 'gender', 'race'] if all(ATTRIBUTE_CONFIG['deepface']['components'].values())
                                   else [k for k, v in ATTRIBUTE_CONFIG['deepface']['components'].items() if v],
                            enforce_detection=False,
                            detector_backend='opencv'
                        )
                        
                        if isinstance(attributes, list):
                            attributes = attributes[0]
                        
                        if ATTRIBUTE_CONFIG['deepface']['components']['age']:
                            result['age'] = attributes.get('age', -1)
                        if ATTRIBUTE_CONFIG['deepface']['components']['gender']:
                            result['gender'] = attributes.get('gender', 'unknown')
                        if ATTRIBUTE_CONFIG['deepface']['components']['race']:
                            result['race'] = attributes.get('dominant_race', 'unknown')
                            result['race_scores'] = attributes.get('race', result['race_scores'])
                            
                    except Exception as e:
                        logger.warning(f"DeepFace analysis failed for {img_path}: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error processing face in {img_path}: {str(e)}")
                result['error'] = f"Face processing error: {str(e)}"
        
        results.extend(batch_results)
    
    return results

def create_debug_visualization(image: np.ndarray, attributes: Dict[str, Any], output_path: str, 
                             landmarks: Dict[str, np.ndarray] = None, nose_bridge: np.ndarray = None) -> None:
    """Create a debug visualization of the processed image with attributes overlaid."""
    # Make a copy of the image
    vis_img = image.copy()
    
    # Draw landmarks if provided
    if landmarks is not None and nose_bridge is not None:
        draw_landmarks(vis_img, landmarks, nose_bridge)
    
    # Image dimensions
    height, width = vis_img.shape[:2]
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(width, height) / 1000  # Scale font based on image size
    thickness = max(1, int(font_scale * 2))
    
    # Background settings for text
    padding = 5
    alpha = 0.5
    
    # Organize attributes into sections
    sections = {
        'Quality': [
            f"Blur: {attributes['quality_metrics']['blur_score']:.2f}",
            f"Bright: {attributes['quality_metrics']['brightness']:.2f}",
            f"Contrast: {attributes['quality_metrics']['contrast']:.2f}",
            f"Compress: {attributes['quality_metrics']['compression_score']:.2f}"
        ],
        'Symmetry': [
            f"Eyes: {attributes['symmetry']['eye_ratio']:.2f}",
            f"Mouth: {attributes['symmetry']['mouth_ratio']:.2f}",
            f"Nose: {attributes['symmetry']['nose_ratio']:.2f}",
            f"Overall: {attributes['symmetry']['overall_symmetry']:.2f}"
        ],
        'Emotions': [
            f"{k}: {v:.2f}" for k, v in attributes['emotion_scores'].items()
        ]
    }
    
    # Position for starting text
    y_pos = padding
    x_margin = 10
    
    for section, lines in sections.items():
        # Draw section header
        text_size = cv2.getTextSize(section, font, font_scale, thickness)[0]
        
        # Create background rectangle for header
        bg_rect = ((x_margin, y_pos), 
                  (x_margin + text_size[0] + 2*padding, 
                   y_pos + text_size[1] + 2*padding))
        overlay = vis_img.copy()
        cv2.rectangle(overlay, bg_rect[0], bg_rect[1], (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
        
        # Draw header text
        cv2.putText(vis_img, section, 
                   (x_margin + padding, y_pos + text_size[1] + padding),
                   font, font_scale, (255, 255, 255), thickness)
        
        y_pos += text_size[1] + 3*padding
        
        # Draw attribute lines
        for line in lines:
            text_size = cv2.getTextSize(line, font, font_scale*0.8, thickness)[0]
            
            # Create background rectangle for line
            bg_rect = ((x_margin + padding, y_pos), 
                      (x_margin + text_size[0] + 3*padding, 
                       y_pos + text_size[1] + padding))
            overlay = vis_img.copy()
            cv2.rectangle(overlay, bg_rect[0], bg_rect[1], (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
            
            # Draw line text
            cv2.putText(vis_img, line, 
                       (x_margin + 2*padding, y_pos + text_size[1]),
                       font, font_scale*0.8, (255, 255, 255), thickness)
            
            y_pos += text_size[1] + 2*padding
        
        y_pos += padding * 2
    
    # Save the visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_img)

def process_dataframe(
    image_paths: List[str],
    batch_size: int = 32,
    debug_vis: bool = False,
    debug_vis_dir: str = None,
    max_debug_images: int = 20
) -> pd.DataFrame:
    """Process all images in a dataframe and return a new dataframe with attributes."""
    # Process images in batches for better memory usage
    print(f"Processing batches:")
    
    batch_size = min(batch_size, 64)  # Cap batch size for stability
    
    # Process images
    all_results = []
    debug_count = 0
    
    # Use tqdm for progress tracking
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_results = process_image_batch(batch_paths, batch_size=batch_size)
        
        # Create debug visualizations if enabled
        if debug_vis and debug_vis_dir and debug_count < max_debug_images:
            for img_path, result in zip(batch_paths, batch_results):
                if debug_count >= max_debug_images:
                    break
                    
                if 'error' not in result or not result['error']:
                    try:
                        # Read original image
                        img = cv2.imread(img_path)
                        if img is not None and '_debug' in result:
                            # Check if debug data is complete and valid
                            debug_data = result.get('_debug', {})
                            if isinstance(debug_data, dict) and 'landmarks' in debug_data and 'nose_bridge' in debug_data:
                                # Create debug visualization with landmarks
                                debug_path = os.path.join(
                                    debug_vis_dir, 
                                    f"debug_{os.path.basename(img_path)}"
                                )
                                create_debug_visualization(
                                    img, result, debug_path,
                                    landmarks=debug_data['landmarks'],
                                    nose_bridge=debug_data['nose_bridge']
                                )
                                debug_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to create debug visualization for {img_path}: {str(e)}")
        
        all_results.extend(batch_results)
        
        # Clear memory periodically
        if i % (batch_size * 10) == 0:
            clear_gpu_memory()
    
    # Convert to DataFrame
    print(f"Converting results to DataFrame... ({len(all_results)} items)")
    results_df = pd.DataFrame(all_results)
    
    # Set image_id as index
    if not results_df.empty and 'image_id' in results_df.columns:
        results_df.set_index('image_id', inplace=True)
    
    return results_df

def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def main():
    """Main function to run the script with command line arguments."""
    parser = argparse.ArgumentParser(description='Generate additional attributes for images')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory of dataset')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to dataset metadata CSV')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save output CSV with attributes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--debug_vis', action='store_true', help='Generate debug visualizations')
    parser.add_argument('--debug_vis_dir', type=str, default=None, help='Directory to save debug visualizations')
    parser.add_argument('--max_debug_images', type=int, default=20, help='Maximum number of debug images to generate')
    parser.add_argument('--disable_deepface', action='store_true', help='Disable DeepFace analysis for faster processing')
    parser.add_argument('--disable_emotions', action='store_true', help='Disable emotion detection for faster processing')
    args = parser.parse_args()
    
    # Update configuration based on CLI arguments
    if args.disable_deepface:
        ATTRIBUTE_CONFIG['deepface']['enabled'] = False
    if args.disable_emotions:
        ATTRIBUTE_CONFIG['emotions']['enabled'] = False
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # Create debug visualization directory if needed
    if args.debug_vis and args.debug_vis_dir:
        os.makedirs(args.debug_vis_dir, exist_ok=True)
    
    # Read metadata
    print(f"Reading metadata from {args.metadata_path}")
    metadata_df = pd.read_csv(args.metadata_path)
    
    # Check for various possible column names for the image path
    image_col = None
    possible_cols = ['filename', 'path', 'Image Path', 'image_path', 'file_path', 'filepath']
    
    for col in possible_cols:
        if col in metadata_df.columns:
            image_col = col
            break
    
    if image_col is None:
        print(f"Error: Could not find image path column in metadata. Available columns: {', '.join(metadata_df.columns)}")
        sys.exit(1)
        
    # Get list of image paths
    image_paths = metadata_df[image_col].tolist()
    if not image_paths:
        print("Error: No image paths found in metadata")
        sys.exit(1)
    
    # Make absolute paths - handle paths correctly based on their format
    processed_paths = []
    for p in image_paths:
        if os.path.isabs(p) and os.path.exists(p):
            # Path is already absolute and exists
            processed_paths.append(p)
        elif p.startswith('/') and not os.path.exists(p):
            # Path starts with / but doesn't exist - treat as relative to data_root
            full_path = os.path.join(args.data_root, p.lstrip('/'))
            processed_paths.append(full_path)
        else:
            # Regular relative path
            full_path = os.path.join(args.data_root, p)
            processed_paths.append(full_path)
    
    image_paths = processed_paths
    
    print(f"Processing {len(image_paths)} images with batch size {args.batch_size}")
    
    # Process images in batches
    results_df = process_dataframe(
        image_paths, 
        batch_size=args.batch_size,
        debug_vis=args.debug_vis,
        debug_vis_dir=args.debug_vis_dir,
        max_debug_images=args.max_debug_images
    )
    
    # Save results
    print(f"Saving results to {args.output_path}")
    results_df.to_csv(args.output_path)
    print("Done!")

if __name__ == "__main__":
    main()
