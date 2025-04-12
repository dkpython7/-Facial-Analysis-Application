import cv2
import mediapipe as mp
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
import time
import tensorflow as tf
from collections import deque, Counter
import json
import base64
from threading import Thread

class FacialAnalysis3D:
    def __init__(self, download_pretrained=True, use_3d=True):
        print("Initializing Enhanced 3D Facial Analysis...")
        # MediaPipe Face Mesh initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True
        )
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self.FACEMESH_CONTOURS = mp.solutions.face_mesh.FACEMESH_CONTOURS
        self.FACEMESH_TESSELATION = mp.solutions.face_mesh.FACEMESH_TESSELATION
        
        # Feature sets
        self.emotions = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'disgust', 'fear']
        self.age_ranges = [
            (0, 2, "Infant"),
            (3, 9, "Child"),
            (10, 19, "Teen"),
            (20, 29, "20s"),
            (30, 39, "30s"),
            (40, 49, "40s"),
            (50, 59, "50s"),
            (60, 69, "60s"),
            (70, 120, "70+")
        ]
        
        # Model directory
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(self.model_dir, exist_ok=True)
        
        if download_pretrained:
            self._download_pretrained_models()
            
        self.emotion_model, self.scaler = self._load_emotion_model()
        self.age_model = self._load_age_model()
        self._init_head_pose_estimation()
        
        # Enhanced emotion smoothing with longer history
        self.emotion_confidence_threshold = 0.4
        self.emotion_history = deque(maxlen=15)
        
        # Age smoothing with moving average
        self.age_history = deque(maxlen=10)
        
        # FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        # 3D visualization settings
        self.use_3d = use_3d
        self.visualization_mode = "full"  # Options: "mesh", "wireframe", "points", "full"
        
        # Enhanced UI parameters
        self.ui_font = cv2.FONT_HERSHEY_SIMPLEX
        self.ui_colors = {
            'box': (0, 255, 0),
            'text_bg': (0, 0, 0, 0.7),  # With alpha for transparency
            'text': (255, 255, 255),
            'axes': [(0, 0, 255), (0, 255, 0), (255, 0, 0)],  # RGB for XYZ
            'neutral': (220, 220, 220),
            'happy': (0, 255, 255),
            'sad': (0, 0, 255),
            'angry': (0, 0, 255),
            'surprised': (255, 255, 0),
            'disgust': (128, 0, 128),
            'fear': (255, 0, 255)
        }
        
        # 3D effect parameters
        self.edge_shader = True
        self.point_size = 2
        self.line_width = 1
        
        # WebServer settings for 3D visualization
        self.web_server_active = False
        self.web_server_port = 8080
        self.web_update_interval = 0.1  # seconds
        
        # Enhanced emotion visualization
        self.show_emotion_color_mapping = True
        self.show_emotion_particles = True
        
        print("Enhanced 3D Facial Analysis initialized successfully!")

    def _download_pretrained_models(self):
        try:
            emotion_model_path = os.path.join(self.model_dir, "emotion_model.pkl")
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            age_model_path = os.path.join(self.model_dir, "age_model.h5")
            
            if not (os.path.exists(emotion_model_path) and os.path.exists(scaler_path)):
                print("Creating emotion models...")
                self._create_fallback_models()
                
            if not os.path.exists(age_model_path):
                print("Creating age model...")
                self._create_fallback_age_model()
        except Exception as e:
            print(f"Error in model setup: {e}")
            self._create_fallback_models()
            self._create_fallback_age_model()

    def _create_fallback_models(self):
        try:
            print("Creating basic fallback emotion models...")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            num_samples = 200
            X = np.random.rand(num_samples, 128)
            y = np.zeros(num_samples)
            samples_per_emotion = num_samples // len(self.emotions)
            
            for i in range(len(self.emotions)):
                start_idx = i * samples_per_emotion
                end_idx = (i + 1) * samples_per_emotion if i < len(self.emotions) - 1 else num_samples
                y[start_idx:end_idx] = i
                
            model.fit(X, y)
            scaler = StandardScaler()
            scaler.fit(X)
            
            with open(os.path.join(self.model_dir, "emotion_model.pkl"), 'wb') as f:
                pickle.dump(model, f)
            with open(os.path.join(self.model_dir, "scaler.pkl"), 'wb') as f:
                pickle.dump(scaler, f)
        except Exception as e:
            print(f"Error creating fallback emotion models: {e}")

    def _create_fallback_age_model(self):
        try:
            print("Creating basic fallback age model...")
            if tf.__version__[0] == '2':
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(64, activation='relu', input_shape=(128,)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                X = np.random.rand(100, 128)
                y = np.random.randint(1, 80, size=(100, 1))
                model.fit(X, y, epochs=5, verbose=0)
                model.save(os.path.join(self.model_dir, "age_model.h5"))
            else:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                X = np.random.rand(100, 128)
                y = np.random.randint(1, 80, size=100)
                model.fit(X, y)
                with open(os.path.join(self.model_dir, "age_model.pkl"), 'wb') as f:
                    pickle.dump(model, f)
        except Exception as e:
            print(f"Error creating fallback age model: {e}")

    def _load_emotion_model(self):
        model_path = os.path.join(self.model_dir, "emotion_model.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print("Loaded pre-trained emotion model and scaler")
            else:
                print("Creating untrained emotion model.")
                model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
                scaler = StandardScaler()
                dummy_features = np.random.rand(10, 128)
                dummy_labels = np.random.randint(0, len(self.emotions), 10)
                model.fit(dummy_features, dummy_labels)
                scaler.fit(dummy_features)
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            scaler = StandardScaler()
            dummy_features = np.random.rand(10, 128)
            dummy_labels = np.random.randint(0, len(self.emotions), 10)
            model.fit(dummy_features, dummy_labels)
            scaler.fit(dummy_features)
            
        return model, scaler

    def _load_age_model(self):
        print(20)
    def _init_head_pose_estimation(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ])
        self.camera_matrix = np.array(
            [[600, 0, 320],
             [0, 600, 240],
             [0, 0, 1]], dtype=np.float64
        )
        self.dist_coeffs = np.zeros((4, 1))

    def _extract_features_from_landmarks(self, landmarks, frame_shape):
        if not isinstance(landmarks, np.ndarray):
            height, width = frame_shape[:2]
            landmarks_array = np.array([(lm.x * width, lm.y * height, lm.z * width)
                                       for lm in landmarks.landmark])
        else:
            landmarks_array = landmarks
            
        features = []
        x_min, y_min = np.min(landmarks_array[:, 0]), np.min(landmarks_array[:, 1])
        x_max, y_max = np.max(landmarks_array[:, 0]), np.max(landmarks_array[:, 1])
        face_width = x_max - x_min
        face_height = y_max - y_min
        
        if face_width == 0 or face_height == 0:
            return np.zeros(128)
            
        norm_landmarks = np.copy(landmarks_array)
        norm_landmarks[:, 0] = (norm_landmarks[:, 0] - x_min) / face_width
        norm_landmarks[:, 1] = (norm_landmarks[:, 1] - y_min) / face_height
        
        # Key points for facial regions
        left_eye = [33, 145, 159, 133, 153, 144]
        right_eye = [362, 380, 374, 263, 385, 386]
        mouth = [61, 39, 37, 0, 267, 269, 291, 405]
        left_eyebrow = [70, 63, 105, 66, 107]
        right_eyebrow = [336, 296, 334, 293, 300]
        nose = [1, 2, 98, 327]
        
        # Extract positions of key points
        for point_set in [left_eye, right_eye, mouth, left_eyebrow, right_eyebrow, nose]:
            for i in point_set:
                if i < len(norm_landmarks):
                    features.extend(norm_landmarks[i, :2])
        
        # Calculate facial feature ratios
        if all(i < len(norm_landmarks) for i in left_eye):
            left_eye_width = np.linalg.norm(norm_landmarks[left_eye[0], :2] - norm_landmarks[left_eye[3], :2])
            left_eye_height = np.linalg.norm(norm_landmarks[left_eye[1], :2] - norm_landmarks[left_eye[5], :2])
            features.append(left_eye_height / (left_eye_width + 1e-6))
        else:
            features.append(0.0)
            
        if all(i < len(norm_landmarks) for i in right_eye):
            right_eye_width = np.linalg.norm(norm_landmarks[right_eye[0], :2] - norm_landmarks[right_eye[3], :2])
            right_eye_height = np.linalg.norm(norm_landmarks[right_eye[1], :2] - norm_landmarks[right_eye[5], :2])
            features.append(right_eye_height / (right_eye_width + 1e-6))
        else:
            features.append(0.0)
            
        # Mouth aspect ratio
        if all(i < len(norm_landmarks) for i in [61, 291, 0, 17]):
            mouth_width = np.linalg.norm(norm_landmarks[61, :2] - norm_landmarks[291, :2])
            mouth_height = np.linalg.norm(norm_landmarks[0, :2] - norm_landmarks[17, :2])
            features.append(mouth_height / (mouth_width + 1e-6))
        else:
            features.append(0.0)
            
        # Eyebrow distances
        if all(i < len(norm_landmarks) for i in [66, 159, 296, 386]):
            left_brow_eye_dist = np.linalg.norm(norm_landmarks[66, :2] - norm_landmarks[159, :2])
            right_brow_eye_dist = np.linalg.norm(norm_landmarks[296, :2] - norm_landmarks[386, :2])
            features.append(left_brow_eye_dist)
            features.append(right_brow_eye_dist)
        else:
            features.extend([0.0, 0.0])
        
        # Calculate angles
        def calculate_angle(p1, p2, p3):
            if not all(i < len(norm_landmarks) for i in [p1, p2, p3]):
                return 0.0
            a = norm_landmarks[p2, :2] - norm_landmarks[p1, :2]
            b = norm_landmarks[p2, :2] - norm_landmarks[p3, :2]
            cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return angle
            
        # Mouth angles
        mouth_angles = [calculate_angle(61, 0, 291), calculate_angle(61, 17, 291)]
        features.extend(mouth_angles)
        
        # Eyebrow curvature
        left_eyebrow_curve = calculate_angle(left_eyebrow[0], left_eyebrow[2], left_eyebrow[4])
        right_eyebrow_curve = calculate_angle(right_eyebrow[0], right_eyebrow[2], right_eyebrow[4])
        features.extend([left_eyebrow_curve, right_eyebrow_curve])
        
        # Mouth corner orientation
        if all(i < len(norm_landmarks) for i in [61, 291, 0]):
            left_corner_angle = math.atan2(
                norm_landmarks[0, 1] - norm_landmarks[61, 1],
                norm_landmarks[0, 0] - norm_landmarks[61, 0]
            )
            right_corner_angle = math.atan2(
                norm_landmarks[0, 1] - norm_landmarks[291, 1],
                norm_landmarks[291, 0] - norm_landmarks[0, 0]
            )
            features.extend([left_corner_angle, right_corner_angle])
        else:
            features.extend([0.0, 0.0])
            
        # Eye openness ratio
        left_eye_open_ratio = left_eye_height / (face_height + 1e-6)
        right_eye_open_ratio = right_eye_height / (face_height + 1e-6)
        features.extend([left_eye_open_ratio, right_eye_open_ratio])
        
        # Ensure exactly 128 features
        features = features[:128]  # Truncate if too many
        if len(features) < 128:
            features.extend([0.0] * (128 - len(features)))  # Pad if too few
            
        return np.array(features)

    def estimate_head_pose(self, landmarks, frame_shape):
        try:
            height, width = frame_shape[:2]
            
            # Define 2D points from facial landmarks
            nose_tip = 1
            chin = 199
            left_eye_corner = 33
            right_eye_corner = 263
            left_mouth = 61
            right_mouth = 291
            
            # Get 2D coordinates
            image_points = np.array([
                (landmarks.landmark[nose_tip].x * width, landmarks.landmark[nose_tip].y * height),
                (landmarks.landmark[chin].x * width, landmarks.landmark[chin].y * height),
                (landmarks.landmark[left_eye_corner].x * width, landmarks.landmark[left_eye_corner].y * height),
                (landmarks.landmark[right_eye_corner].x * width, landmarks.landmark[right_eye_corner].y * height),
                (landmarks.landmark[left_mouth].x * width, landmarks.landmark[left_mouth].y * height),
                (landmarks.landmark[right_mouth].x * width, landmarks.landmark[right_mouth].y * height)
            ], dtype=np.float64)
            
            # Solve PnP problem
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, self.camera_matrix, self.dist_coeffs)
                
            if not success:
                return "Center", (0, 0, 0), rotation_vector, translation_vector
                
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract Euler angles
            euler_angles = self._rotation_matrix_to_euler_angles(rotation_matrix) * 180 / np.pi
            
            # Determine head pose based on angles
            pose = ""
            
            # Yaw (looking left/right)
            yaw = euler_angles[1]
            if yaw < -15:
                pose += "Right "
            elif yaw > 15:
                pose += "Left "
                
            # Pitch (looking up/down)
            pitch = euler_angles[0]
            if pitch < -15:
                pose += "Up"
            elif pitch > 15:
                pose += "Down"
                
            # Roll (tilting head)
            roll = euler_angles[2]
            if abs(roll) > 15 and not pose:
                if roll < 0:
                    pose += "Tilted Right"
                else:
                    pose += "Tilted Left"
                    
            # If no direction is detected, the head is centered
            if not pose:
                pose = "Center"
                
            return pose, euler_angles, rotation_vector, translation_vector
        except Exception as e:
            print(f"Head pose estimation error: {e}")
            return "Center", (0, 0, 0), np.zeros((3, 1)), np.zeros((3, 1))

    def _rotation_matrix_to_euler_angles(self, R):
        # Check if rotation matrix has a gimbal lock
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])  # Pitch
            y = math.atan2(-R[2, 0], sy)      # Yaw
            z = math.atan2(R[1, 0], R[0, 0])  # Roll
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
            
        return np.array([x, y, z])

    def detect_emotion(self, landmarks, frame_shape):
        try:
            # Extract features
            features = self._extract_features_from_landmarks(landmarks, frame_shape)
            
            try:
                # Scale features and predict emotion
                features_scaled = self.scaler.transform([features])
                emotion_idx = self.emotion_model.predict(features_scaled)[0]
                probs = self.emotion_model.predict_proba(features_scaled)[0]
                max_prob = np.max(probs)
                
                # Get emotion with confidence checking
                if isinstance(emotion_idx, (int, np.integer)) and 0 <= emotion_idx < len(self.emotions):
                    if max_prob < self.emotion_confidence_threshold:
                        emotion = self._rule_based_emotion(landmarks, frame_shape)
                    else:
                        emotion = self.emotions[emotion_idx]
                else:
                    emotion = self._rule_based_emotion(landmarks, frame_shape)
                    
                # Add to history for improved smoothing
                self.emotion_history.append(emotion)
                
                # Get most common emotion from history for stability
                if self.emotion_history:
                    emotion_counts = Counter(self.emotion_history)
                    # Apply weighted count - recent emotions count more
                    for i, e in enumerate(self.emotion_history):
                        emotion_counts[e] += i / len(self.emotion_history)
                    emotion = emotion_counts.most_common(1)[0][0]
                    
            except Exception as e:
                print(f"ML emotion detection error: {e}")
                emotion = self._rule_based_emotion(landmarks, frame_shape)
                
            return emotion
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return "neutral"

    def _rule_based_emotion(self, landmarks, frame_shape):
        try:
            height, width = frame_shape[:2]
            
            # Eyes
            left_eye_top = 159
            left_eye_bottom = 145
            right_eye_top = 386
            right_eye_bottom = 374
            
            # Mouth
            mouth_left = 61
            mouth_right = 291
            mouth_top = 0
            mouth_bottom = 17
            
            # Eyebrows
            left_eyebrow_outer = 70
            left_eyebrow_center = 66
            left_eyebrow_inner = 107
            right_eyebrow_outer = 336
            right_eyebrow_center = 296
            right_eyebrow_inner = 334
            
            # Get landmark coordinates and calculate metrics
            left_eye_h = landmarks.landmark[left_eye_bottom].y - landmarks.landmark[left_eye_top].y
            right_eye_h = landmarks.landmark[right_eye_bottom].y - landmarks.landmark[right_eye_top].y
            mouth_w = landmarks.landmark[mouth_right].x - landmarks.landmark[mouth_left].x
            mouth_h = landmarks.landmark[mouth_bottom].y - landmarks.landmark[mouth_top].y
            
            # Face size for normalization
            face_height = max([lm.y for lm in landmarks.landmark]) - min([lm.y for lm in landmarks.landmark])
            face_width = max([lm.x for lm in landmarks.landmark]) - min([lm.x for lm in landmarks.landmark])
            
            # Normalize metrics
            eye_openness = (left_eye_h + right_eye_h) / 2
            eye_openness_norm = eye_openness / face_height
            mouth_aspect_ratio_norm = mouth_h / (mouth_w + 1e-6)
            
            # Eyebrow positions
            left_eyebrow_eye_dist = landmarks.landmark[left_eyebrow_center].y - landmarks.landmark[left_eye_top].y
            right_eyebrow_eye_dist = landmarks.landmark[right_eyebrow_center].y - landmarks.landmark[right_eye_top].y
            
            left_eyebrow_eye_dist_norm = left_eyebrow_eye_dist / face_height
            right_eyebrow_eye_dist_norm = right_eyebrow_eye_dist / face_height
            
            # Eyebrow angle
            left_eyebrow_angle = math.atan2(
                landmarks.landmark[left_eyebrow_inner].y - landmarks.landmark[left_eyebrow_outer].y,
                landmarks.landmark[left_eyebrow_inner].x - landmarks.landmark[left_eyebrow_outer].x
            )
            right_eyebrow_angle = math.atan2(
                landmarks.landmark[right_eyebrow_inner].y - landmarks.landmark[right_eyebrow_outer].y,
                landmarks.landmark[right_eyebrow_inner].x - landmarks.landmark[right_eyebrow_outer].x
            )
            
            # Mouth corners
            mouth_corner_left_angle = math.atan2(
                landmarks.landmark[mouth_left].y - landmarks.landmark[mouth_top].y,
                landmarks.landmark[mouth_left].x - landmarks.landmark[mouth_top].x
            )
            mouth_corner_right_angle = math.atan2(
                landmarks.landmark[mouth_right].y - landmarks.landmark[mouth_top].y,
                landmarks.landmark[mouth_right].x - landmarks.landmark[mouth_top].x
            )
            
            # Decision logic for emotions
            # Happy: Mouth corners up, wide mouth
            if (mouth_corner_left_angle < -0.1 and mouth_corner_right_angle > 0.1 and
                    mouth_aspect_ratio_norm > 0.25):
                return "happy"
            # Sad: Mouth corners down, eyebrows angled
            elif (mouth_corner_left_angle > 0.1 and mouth_corner_right_angle < -0.1 and
                    left_eyebrow_angle < 0 and right_eyebrow_angle > 0):
                return "sad"
            # Angry: Eyebrows lowered and angled inward
            elif (left_eyebrow_eye_dist_norm < 0.05 and right_eyebrow_eye_dist_norm < 0.05 and
                    left_eyebrow_angle < -0.2 and right_eyebrow_angle > 0.2):
                return "angry"
            # Surprised: Eyes wide open, eyebrows raised, mouth open
            elif (eye_openness_norm > 0.07 and
                    left_eyebrow_eye_dist_norm > 0.08 and right_eyebrow_eye_dist_norm > 0.08 and
                    mouth_aspect_ratio_norm > 0.4):
                return "surprised"
            # Fearful: Eyes wide, eyebrows raised, mouth slightly open
            elif (eye_openness_norm > 0.07 and
                    left_eyebrow_eye_dist_norm > 0.08 and right_eyebrow_eye_dist_norm > 0.08 and
                    mouth_aspect_ratio_norm > 0.2 and mouth_aspect_ratio_norm < 0.4):
                return "fear"
            # Disgust: Eyes narrowed, eyebrows lowered, mouth angled
            elif (eye_openness_norm < 0.05 and
                    left_eyebrow_eye_dist_norm < 0.05 and
                    abs(mouth_corner_left_angle - mouth_corner_right_angle) > 0.3):
                return "disgust"
            # Default to neutral
            else:
                return "neutral"
        except Exception as e:
            print(f"Rule-based emotion detection error: {e}")
            return "neutral"

    def estimate_age(self, face_landmarks, frame_shape, face_image=None):
        try:
            # Extract features from landmarks
            features = self._extract_features_from_landmarks(face_landmarks, frame_shape)
            
            # Use the appropriate model type
            if self.age_model["type"] == "tensorflow":
                features_reshaped = features.reshape(1, -1)
                predicted_age = self.age_model["model"].predict(features_reshaped)[0][0]
            elif self.age_model["type"] == "pickle":
                predicted_age = self.age_model["model"].predict([features])[0]
            else:  # Function
                predicted_age = self.age_model["model"](features)
            
            # Apply smoothing using moving average
            self.age_history.append(predicted_age)
            smoothed_age = np.mean(self.age_history)
            
            # Determine age range category
            age_category = None
            for min_age, max_age, category in self.age_ranges:
                if min_age <= smoothed_age <= max_age:
                    age_category = category
                    break
            
            if age_category is None:
                age_category = "Adult"
                
            return int(smoothed_age), age_category
        except Exception as e:
            print(f"Age estimation error: {e}")
            return 20, "Adult"

    def process_frame(self, frame, visualize=True):
        """Process a frame to extract facial features and analyze them."""
        try:
            # Calculate FPS
            self.new_frame_time = time.time()
            fps = 1 / (self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 30
            self.prev_frame_time = self.new_frame_time
            
            # Create a copy for visualization
            if visualize:
                viz_frame = frame.copy()
            else:
                viz_frame = None
                
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                if visualize:
                    self._draw_fps(viz_frame, fps)
                    return viz_frame, None
                return frame, None
                
            face_landmarks = results.multi_face_landmarks[0]
            frame_height, frame_width = frame.shape[:2]
            frame_shape = (frame_height, frame_width)
            
            # Extract facial analysis data
            emotion = self.detect_emotion(face_landmarks, frame_shape)
            head_pose, euler_angles, rot_vec, trans_vec = self.estimate_head_pose(face_landmarks, frame_shape)
            age, age_category = self.estimate_age(face_landmarks, frame_shape)
            
            # Combine results
            facial_data = {
                'emotion': emotion,
                'head_pose': head_pose,
                'age': age,
                'age_category': age_category,
                'euler_angles': euler_angles,
                'fps': fps,
                'landmarks': face_landmarks
            }
            
            # Visualize results
            if visualize:
                viz_frame = self._visualize_results(viz_frame, facial_data)
                return viz_frame, facial_data
                
            return frame, facial_data
        except Exception as e:
            print(f"Frame processing error: {e}")
            if visualize:
                return frame, None
            return frame, None

    def _visualize_results(self, frame, facial_data):
        """Visualize the facial analysis results on the frame."""
        try:
            if facial_data is None:
                return frame
                
            height, width = frame.shape[:2]
            landmarks = facial_data['landmarks']
            emotion = facial_data['emotion']
            head_pose = facial_data['head_pose']
            age = facial_data['age']
            age_category = facial_data['age_category']
            euler_angles = facial_data['euler_angles']
            fps = facial_data['fps']
            
            # Choose visualization mode
            if self.visualization_mode == "mesh":
                self._draw_face_mesh(frame, landmarks)
            elif self.visualization_mode == "wireframe":
                self._draw_face_wireframe(frame, landmarks)
            elif self.visualization_mode == "points":
                self._draw_face_points(frame, landmarks)
            else:  # "full" - both mesh and stats
                self._draw_face_mesh(frame, landmarks)
                
            # Draw head pose
            self._draw_head_pose(frame, landmarks, rot_vec=None, trans_vec=None, euler_angles=euler_angles)
            
            # Draw emotion
            self._draw_emotion(frame, emotion)
            
            # Draw age
            self._draw_age(frame, age, age_category)
            
            # Draw FPS
            self._draw_fps(frame, fps)
            
            return frame
        except Exception as e:
            print(f"Visualization error: {e}")
            return frame

    def _draw_face_mesh(self, frame, landmarks):
        """Draw the face mesh on the frame."""
        try:
            height, width = frame.shape[:2]
            
            # Draw tesselation
            connections = self.FACEMESH_TESSELATION
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = (int(landmarks.landmark[start_idx].x * width), 
                               int(landmarks.landmark[start_idx].y * height))
                end_point = (int(landmarks.landmark[end_idx].x * width), 
                             int(landmarks.landmark[end_idx].y * height))
                
                # Apply 3D effect with depth-based coloring
                z_start = landmarks.landmark[start_idx].z
                z_end = landmarks.landmark[end_idx].z
                z_avg = (z_start + z_end) / 2
                
                # Map z to color (closer is warmer/red, farther is cooler/blue)
                z_norm = (z_avg + 0.1) * 5  # Normalize z value
                z_norm = max(0, min(1, z_norm))  # Clamp to [0, 1]
                
                # Color gradient based on depth
                color = (
                    int(255 * (1 - z_norm)),  # Blue
                    0,                         # Green
                    int(255 * z_norm)          # Red
                )
                
                # Draw the connection with depth-appropriate thickness
                thickness = max(1, int(2 * (1 - abs(z_avg))))
                cv2.line(frame, start_point, end_point, color, thickness)
                
            # Draw contours with different color
            connections = self.FACEMESH_CONTOURS
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = (int(landmarks.landmark[start_idx].x * width), 
                               int(landmarks.landmark[start_idx].y * height))
                end_point = (int(landmarks.landmark[end_idx].x * width), 
                             int(landmarks.landmark[end_idx].y * height))
                
                cv2.line(frame, start_point, end_point, (0, 255, 255), 2)
                
        except Exception as e:
            print(f"Face mesh drawing error: {e}")

    def _draw_face_wireframe(self, frame, landmarks):
        """Draw a simplified wireframe of the face."""
        try:
            height, width = frame.shape[:2]
            
            # Define key facial feature connections
            connections = self.FACEMESH_CONTOURS
            
            # Draw connections
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_point = (int(landmarks.landmark[start_idx].x * width), 
                               int(landmarks.landmark[start_idx].y * height))
                end_point = (int(landmarks.landmark[end_idx].x * width), 
                             int(landmarks.landmark[end_idx].y * height))
                
                cv2.line(frame, start_point, end_point, (0, 255, 0), self.line_width)
                
        except Exception as e:
            print(f"Wireframe drawing error: {e}")

    def _draw_face_points(self, frame, landmarks):
        """Draw facial landmark points."""
        try:
            height, width = frame.shape[:2]
            
            for i, landmark in enumerate(landmarks.landmark):
                # Scale to image size
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                z = landmark.z
                
                # Use depth for point size
                point_size = max(1, int(self.point_size * (1 + abs(z))))
                
                # Use depth for color: closer points are warmer
                z_norm = (z + 0.1) * 5  # Normalize z value
                z_norm = max(0, min(1, z_norm))  # Clamp to [0, 1]
                
                # Color based on facial feature type
                if i in range(0, 17):  # Face outline
                    color = (0, 255, 0)  # Green
                elif i in range(17, 78):  # Eyebrows and nose
                    color = (255, 0, 0)  # Blue
                elif i in range(78, 134):  # Eyes
                    color = (0, 0, 255)  # Red
                elif i in range(134, 155):  # Inner mouth
                    color = (0, 255, 255)  # Yellow
                else:  # Other features
                    color = (255, 255, 0)  # Cyan
                    
                cv2.circle(frame, (x, y), point_size, color, -1)
                
        except Exception as e:
            print(f"Points drawing error: {e}")

    def _draw_head_pose(self, frame, landmarks, rot_vec=None, trans_vec=None, euler_angles=None):
        """Draw head pose information and axes."""
        try:
            height, width = frame.shape[:2]
            
            # Draw nose position for reference
            nose_tip = (int(landmarks.landmark[1].x * width), 
                        int(landmarks.landmark[1].y * height))
            
            if euler_angles is not None:
                # Draw text with Euler angles
                angles_text = f"Pitch: {euler_angles[0]:.1f}, Yaw: {euler_angles[1]:.1f}, Roll: {euler_angles[2]:.1f}"
                
                # Draw background for better readability
                text_size = cv2.getTextSize(angles_text, self.ui_font, 0.5, 1)[0]
                cv2.rectangle(frame, 
                             (10, height - 45), 
                             (10 + text_size[0], height - 25), 
                             (0, 0, 0), 
                             -1)
                             
                cv2.putText(frame, 
                           angles_text, 
                           (10, height - 30), 
                           self.ui_font, 
                           0.5, 
                           (255, 255, 255), 
                           1)
                
                # Draw 3D axes at nose position
                axis_length = 50
                pitch, yaw, roll = euler_angles
                
                # Convert Euler angles to rotation matrix
                Rx = np.array([
                    [1, 0, 0],
                    [0, math.cos(pitch * math.pi / 180), -math.sin(pitch * math.pi / 180)],
                    [0, math.sin(pitch * math.pi / 180), math.cos(pitch * math.pi / 180)]
                ])
                
                Ry = np.array([
                    [math.cos(yaw * math.pi / 180), 0, math.sin(yaw * math.pi / 180)],
                    [0, 1, 0],
                    [-math.sin(yaw * math.pi / 180), 0, math.cos(yaw * math.pi / 180)]
                ])
                
                Rz = np.array([
                    [math.cos(roll * math.pi / 180), -math.sin(roll * math.pi / 180), 0],
                    [math.sin(roll * math.pi / 180), math.cos(roll * math.pi / 180), 0],
                    [0, 0, 1]
                ])
                
                # Combine rotation matrices
                R = Rx @ Ry @ Rz
                
                # Define axis vectors
                axes = np.array([
                    [axis_length, 0, 0],  # X-axis (red)
                    [0, axis_length, 0],  # Y-axis (green)
                    [0, 0, axis_length]   # Z-axis (blue)
                ])
                
                # Project axes to 2D
                for i, axis in enumerate(axes):
                    # Rotate axis by head rotation
                    rotated_axis = R @ axis
                    
                    # Project to 2D image plane (simple perspective projection)
                    x = int(nose_tip[0] + rotated_axis[0])
                    y = int(nose_tip[1] - rotated_axis[1])  # Negative Y because image coordinates go down
                    
                    # Draw axis line
                    cv2.line(frame, nose_tip, (x, y), self.ui_colors['axes'][i], 2)
                    
                    # Draw axis label
                    axis_labels = ['X', 'Y', 'Z']
                    cv2.putText(frame, axis_labels[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ui_colors['axes'][i], 2)
                    
        except Exception as e:
            print(f"Head pose visualization error: {e}")

    def _draw_emotion(self, frame, emotion):
        """Draw emotion text with color coding."""
        try:
            height, width = frame.shape[:2]
            
            # Get color for emotion
            if emotion in self.ui_colors:
                color = self.ui_colors[emotion]
            else:
                color = (255, 255, 255)
                
            # Create text
            emotion_text = f"Emotion: {emotion.capitalize()}"
            
            # Draw background for text
            text_size = cv2.getTextSize(emotion_text, self.ui_font, 0.6, 2)[0]
            cv2.rectangle(frame, 
                         (10, 10), 
                         (10 + text_size[0], 10 + text_size[1] + 10), 
                         (0, 0, 0), 
                         -1)
                         
            # Draw text
            cv2.putText(frame, 
                       emotion_text, 
                       (10, 30), 
                       self.ui_font, 
                       0.6, 
                       color, 
                       2)
                       
            # Add emotion-specific particles for enhanced visualization
            if self.show_emotion_particles:
                self._draw_emotion_particles(frame, emotion)
                
        except Exception as e:
            print(f"Emotion visualization error: {e}")

    def _draw_emotion_particles(self, frame, emotion):
        """Draw emotion-specific particle effects."""
        try:
            height, width = frame.shape[:2]
            
            if emotion == "happy":
                # Draw sparkles around the frame
                num_particles = 10
                for _ in range(num_particles):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height // 3)
                    size = np.random.randint(2, 5)
                    cv2.circle(frame, (x, y), size, self.ui_colors[emotion], -1)
                    
            elif emotion == "sad":
                # Draw falling tear-like particles
                num_particles = 5
                for i in range(num_particles):
                    x = width // 2 + np.random.randint(-50, 50)
                    y = height // 3 + i * height // 10
                    cv2.circle(frame, (x, y), 3, self.ui_colors[emotion], -1)
                    
            elif emotion == "angry":
                # Draw red lightning bolts
                num_bolts = 3
                for _ in range(num_bolts):
                    start_x = np.random.randint(width // 4, width * 3 // 4)
                    start_y = 0
                    points = [(start_x, start_y)]
                    
                    x, y = start_x, start_y
                    for _ in range(3):
                        x += np.random.randint(-20, 20)
                        y += height // 6
                        points.append((x, y))
                        
                    for i in range(len(points) - 1):
                        cv2.line(frame, points[i], points[i+1], self.ui_colors[emotion], 2)
                        
            elif emotion == "surprised":
                # Draw circles expanding outward
                center_x, center_y = width // 2, height // 2
                num_circles = 3
                for i in range(num_circles):
                    radius = (i + 1) * 20
                    cv2.circle(frame, (center_x, center_y), radius, self.ui_colors[emotion], 1)
                    
        except Exception as e:
            print(f"Emotion particles visualization error: {e}")

    def _draw_age(self, frame, age, age_category):
        """Draw age information."""
        try:
            height = frame.shape[0]
            
            # Create text
            age_text = f"Age: {age} ({age_category})"
            
            # Draw background for text
            text_size = cv2.getTextSize(age_text, self.ui_font, 0.6, 2)[0]
            cv2.rectangle(frame, 
                         (10, height - 20 - text_size[1]), 
                         (10 + text_size[0], height - 10), 
                         (0, 0, 0), 
                         -1)
                         
            # Draw text
            cv2.putText(frame, 
                       age_text, 
                       (10, height - 15), 
                       self.ui_font, 
                       0.6, 
                       (255, 255, 255), 
                       2)
                       
        except Exception as e:
            print(f"Age visualization error: {e}")

    def _draw_fps(self, frame, fps):
        """Draw FPS counter."""
        try:
            # Create text
            fps_text = f"FPS: {fps:.1f}"
            
            # Draw background for text
            text_size = cv2.getTextSize(fps_text, self.ui_font, 0.5, 1)[0]
            cv2.rectangle(frame, 
                         (frame.shape[1] - 10 - text_size[0], 10), 
                         (frame.shape[1] - 10, 10 + text_size[1] + 10), 
                         (0, 0, 0), 
                         -1)
                         
            # Draw text
            cv2.putText(frame, 
                       fps_text, 
                       (frame.shape[1] - 10 - text_size[0], 30), 
                       self.ui_font, 
                       0.5, 
                       (0, 255, 0), 
                       1)
                       
        except Exception as e:
            print(f"FPS visualization error: {e}")

    def start_3d_visualization_server(self):
        """Start a web server for 3D visualization if not already running."""
        if not self.web_server_active:
            try:
                self.web_server_active = True
                thread = Thread(target=self._run_3d_web_server)
                thread.daemon = True
                thread.start()
                print(f"3D visualization server started on port {self.web_server_port}")
                return True
            except Exception as e:
                print(f"Failed to start 3D server: {e}")
                self.web_server_active = False
                return False
        return True

    def _run_3d_web_server(self):
        """Run a simple web server for 3D visualization using Three.js."""
        try:
            import http.server
            import socketserver
            
            # Create a simple HTML page with Three.js for 3D visualization
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>3D Facial Analysis Visualization</title>
                <style>
                    body { margin: 0; overflow: hidden; }
                    canvas { display: block; }
                    #info {
                        position: absolute;
                        top: 10px;
                        left: 10px;
                        color: white;
                        background-color: rgba(0,0,0,0.7);
                        padding: 10px;
                        border-radius: 5px;
                    }
                </style>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.min.js"></script>
            </head>
            <body>
                <div id="info">
                    <h2>3D Facial Analysis</h2>
                    <div id="emotion">Emotion: --</div>
                    <div id="age">Age: --</div>
                    <div id="head-pose">Head Pose: --</div>
                </div>
                <script>
                    // Initialize Three.js scene
                    const scene = new THREE.Scene();
                    scene.background = new THREE.Color(0x000000);
                    
                    // Camera setup
                    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                    camera.position.z = 50;
                    
                    // Renderer
                    const renderer = new THREE.WebGLRenderer({ antialias: true });
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    document.body.appendChild(renderer.domElement);
                    
                    // Add lights
                    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
                    scene.add(ambientLight);
                    
                    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                    directionalLight.position.set(0, 1, 1);
                    scene.add(directionalLight);
                    
                    // Face mesh group
                    const faceGroup = new THREE.Group();
                    scene.add(faceGroup);
                    
                    // Settings
                    const settings = {
                        visualizationMode: 'full',
                        pointSize: 0.2,
                        lineWidth: 0.1,
                        showEmotionColor: true,
                        rotateModel: true
                    };
                    
                    // GUI Controls
                    const gui = new dat.GUI();
                    gui.add(settings, 'visualizationMode', ['full', 'wireframe', 'points']).onChange(updateVisualization);
                    gui.add(settings, 'pointSize', 0.1, 1).onChange(updateVisualization);
                    gui.add(settings, 'lineWidth', 0.1, 1).onChange(updateVisualization);
                    gui.add(settings, 'showEmotionColor');
                    gui.add(settings, 'rotateModel');
                    
                    // Emotion colors
                    const emotionColors = {
                        'neutral': 0xdcdcdc,
                        'happy': 0x00ffff,
                        'sad': 0x0000ff,
                        'angry': 0xff0000,
                        'surprised': 0xffff00,
                        'disgust': 0x800080,
                        'fear': 0xff00ff
                    };
                    
                    // Update with new data
                    function updateFaceMesh(landmarks, emotion, age, headPose) {
                        // Clear previous mesh
                        while(faceGroup.children.length > 0) { 
                            faceGroup.remove(faceGroup.children[0]); 
                        }
                        
                        // Update info display
                        document.getElementById('emotion').textContent = `Emotion: ${emotion}`;
                        document.getElementById('age').textContent = `Age: ${age}`;
                        document.getElementById('head-pose').textContent = `Head Pose: ${headPose}`;
                        
                        // Create material based on emotion
                        const color = emotionColors[emotion] || 0xffffff;
                        const material = new THREE.MeshStandardMaterial({ 
                            color: settings.showEmotionColor ? color : 0xffffff,
                            emissive: settings.showEmotionColor ? color : 0xffffff,
                            emissiveIntensity: 0.2,
                            metalness: 0.8,
                            roughness: 0.2
                        });
                        
                        const pointsMaterial = new THREE.PointsMaterial({ 
                            color: settings.showEmotionColor ? color : 0xffffff,
                            size: settings.pointSize 
                        });
                        
                        // Create geometry
                        const geometry = new THREE.BufferGeometry();
                        const positions = [];
                        const indices = [];
                        
                        // Add vertices
                        landmarks.forEach(point => {
                            // Scale and flip coordinates as needed
                            positions.push(point.x * 100 - 50);  // Center X
                            positions.push(-(point.y * 100 - 50));  // Flip Y
                            positions.push(-point.z * 300);  // Scale Z
                        });
                        
                        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                        
                        // Add connections
                        const connections = getFaceMeshConnections();
                        connections.forEach(connection => {
                            indices.push(connection[0], connection[1]);
                        });
                        
                        geometry.setIndex(indices);
                        
                        // Create mesh based on visualization mode
                        if (settings.visualizationMode === 'points' || settings.visualizationMode === 'full') {
                            const points = new THREE.Points(geometry.clone(), pointsMaterial);
                            faceGroup.add(points);
                        }
                        
                        if (settings.visualizationMode === 'wireframe' || settings.visualizationMode === 'full') {
                            const lines = new THREE.LineSegments(
                                geometry, 
                                new THREE.LineBasicMaterial({ 
                                    color: settings.showEmotionColor ? color : 0xffffff, 
                                    linewidth: settings.lineWidth 
                                })
                            );
                            faceGroup.add(lines);
                        }
                    }
                    
                    function getFaceMeshConnections() {
                        // Simplified face mesh connections
                        return [
                            // Face outline
                            [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389],
                            [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397],
                            [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152],
                            [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172],
                            [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162],
                            [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
                            
                            // Eyes
                            [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154],
                            [154, 155], [155, 133], [33, 246], [246, 161], [161, 160], [160, 159],
                            [159, 158], [158, 157], [157, 173], [173, 133],
                            
                            [263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381],
                            [381, 382], [382, 362], [263, 466], [466, 388], [388, 387], [387, 386],
                            [386, 385], [385, 384], [384, 398], [398, 362],
                            
                            // Eyebrows
                            [70, 63], [63, 105], [105, 66], [66, 107], [107, 55], [55, 65],
                            [336, 296], [296, 334], [334, 293], [293, 300], [300, 285], [285, 295],
                            
                            // Nose
                            [168, 6], [6, 197], [197, 195], [195, 5], [5, 4], [4, 1], [1, 19], [19, 94],
                            [94, 2], [2, 164], [164, 0], [0, 9], [9, 151], [151, 10],
                            
                            // Mouth
                            [61, 185], [185, 40], [40, 39], [39, 37], [37, 0], [0, 267],
                            [267, 269], [269, 270], [270, 409], [409, 291], [61, 146], [146, 91],
                            [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 321],
                            [321, 375], [375, 291]
                        ];
                    }
                    
                    function updateVisualization() {
                        // This will be called when settings change
                        // The next data update will apply the new settings
                    }
                    
                    // Animation loop
                    function animate() {
                        requestAnimationFrame(animate);
                        
                        // Rotate model if enabled
                        if (settings.rotateModel) {
                            faceGroup.rotation.y += 0.005;
                        }
                        
                        renderer.render(scene, camera);
                    }
                    
                    // Handle window resize
                    window.addEventListener('resize', () => {
                        camera.aspect = window.innerWidth / window.innerHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize(window.innerWidth, window.innerHeight);
                    });
                    
                    // Poll for updates
                    setInterval(fetchFacialData, 100);
                    
                    function fetchFacialData() {
                        fetch('/facial-data')
                            .then(response => response.json())
                            .then(data => {
                                if (data && data.landmarks) {
                                    updateFaceMesh(
                                        data.landmarks, 
                                        data.emotion, 
                                        data.age + " (" + data.age_category + ")", 
                                        data.head_pose
                                    );
                                }
                            })
                            .catch(err => console.error('Error fetching data:', err));
                    }
                    
                    // Start animation
                    animate();
                </script>
            </body>
            </html>
            """
            
            # Store the latest facial data
            facial_data_json = "{}"
            
            class FacialDataHandler(http.server.SimpleHTTPRequestHandler):
                def do_GET(self):
                    nonlocal facial_data_json
                    
                    if self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(html.encode())
                    elif self.path == '/facial-data':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(facial_data_json.encode())
                    else:
                        self.send_error(404)
                        
                def log_message(self, format, *args):
                    # Suppress log messages
                    return
            
            # Update facial data periodically
            def update_facial_data():
                nonlocal facial_data_json
                while self.web_server_active:
                    # Get the latest facial data
                    if hasattr(self, 'latest_facial_data') and self.latest_facial_data:
                        try:
                            # Convert landmarks to serializable format
                            landmarks = []
                            for i, lm in enumerate(self.latest_facial_data['landmarks'].landmark):
                                landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z})
                                
                            data = {
                                'landmarks': landmarks,
                                'emotion': self.latest_facial_data['emotion'],
                                'age': self.latest_facial_data['age'],
                                'age_category': self.latest_facial_data['age_category'],
                                'head_pose': self.latest_facial_data['head_pose'],
                                'euler_angles': self.latest_facial_data['euler_angles'].tolist()
                            }
                            
                            facial_data_json = json.dumps(data)
                        except Exception as e:
                            print(f"Error serializing facial data: {e}")
                    
                    time.sleep(self.web_update_interval)
            
            # Start update thread
            update_thread = Thread(target=update_facial_data)
            update_thread.daemon = True
            update_thread.start()
            
            # Start web server
            with socketserver.TCPServer(("", self.web_server_port), FacialDataHandler) as httpd:
                print(f"Serving 3D visualization at port {self.web_server_port}")
                while self.web_server_active:
                    httpd.handle_request()
                    
        except Exception as e:
            print(f"Web server error: {e}")
            self.web_server_active = False

    def update_latest_facial_data(self, facial_data):
        """Update the latest facial data for the web visualization."""
        self.latest_facial_data = facial_data
        
    def set_visualization_mode(self, mode):
        """Set the visualization mode."""
        valid_modes = ["mesh", "wireframe", "points", "full"]
        if mode in valid_modes:
            self.visualization_mode = mode
            return True
        return False
        
    def export_data_to_json(self, facial_data, filename=None):
        """Export the facial analysis data to JSON."""
        try:
            if facial_data is None:
                return False, "No facial data available"
                
            # Convert landmarks to serializable format
            landmarks = []
            for i, lm in enumerate(facial_data['landmarks'].landmark):
                landmarks.append({
                    'index': i,
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z
                })
                
            data = {
                'timestamp': time.time(),
                'emotion': facial_data['emotion'],
                'age': facial_data['age'],
                'age_category': facial_data['age_category'],
                'head_pose': facial_data['head_pose'],
                'euler_angles': facial_data['euler_angles'].tolist(),
                'landmarks': landmarks
            }
            
            if filename is None:
                timestamp = int(time.time())
                filename = f"facial_data_{timestamp}.json"
                
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
            return True, filename
        except Exception as e:
            return False, f"Export error: {e}"
            
    def frame_to_base64(self, frame):
        """Convert a frame to base64 encoding for web display."""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Frame encoding error: {e}")
            return ""

    def enhance_visualization(self, frame, facial_data):
        """Apply enhanced visualization effects."""
        try:
            if facial_data is None:
                return frame
                
            # Add a subtle vignette effect
            height, width = frame.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            center = (width // 2, height // 2)
            max_radius = min(center[0], center[1])
            
            cv2.circle(mask, center, max_radius, 255, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), max_radius // 3)
            mask = mask.astype(float) / 255
            mask = np.expand_dims(mask, axis=2)
            mask = np.repeat(mask, 3, axis=2)
            
            # Apply vignette
            frame = frame * mask + frame * (1 - mask) * 0.6
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            
            # Add emotion color overlay
            emotion = facial_data['emotion']
            if emotion in self.ui_colors:
                color_overlay = np.zeros_like(frame, dtype=np.float32)
                color_overlay[:, :] = self.ui_colors[emotion]
                
                # Apply subtle color overlay
                alpha = 0.1
                frame = cv2.addWeighted(frame, 1 - alpha, color_overlay.astype(np.uint8), alpha, 0)
                
            return frame
        except Exception as e:
            print(f"Enhanced visualization error: {e}")
            return frame

    def run_camera(self, camera_id=0, window_name="3D Facial Analysis", resolution=(220, 140)):
        """Run the facial analysis on a camera feed with interactive controls."""
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            
            if not cap.isOpened():
                print("Error: Could not open camera.")
                return
                
            # Start 3D web server if enabled
            if self.use_3d:
                self.start_3d_visualization_server()
                
            print("\nControls:")
            print("  M - Change visualization mode")
            print("  3 - Toggle 3D web visualization")
            print("  E - Toggle emotion color")
            print("  S - Save current frame and data")
            print("  Q or ESC - Quit")
            
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                    
                # Process frame
                processed_frame, facial_data = self.process_frame(frame)
                
                # Update web visualization data
                if self.web_server_active and facial_data:
                    self.update_latest_facial_data(facial_data)
                    
                # Apply enhanced visualization
                if facial_data:
                    processed_frame = self.enhance_visualization(processed_frame, facial_data)
                    
                # Show frame
                cv2.imshow(window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('m'):  # Change visualization mode
                    modes = ["mesh", "wireframe", "points", "full"]
                    current_idx = modes.index(self.visualization_mode)
                    next_idx = (current_idx + 1) % len(modes)
                    self.visualization_mode = modes[next_idx]
                    print(f"Visualization mode: {self.visualization_mode}")
                elif key == ord('3'):  # Toggle 3D visualization
                    if self.web_server_active:
                        self.web_server_active = False
                        print("3D visualization disabled")
                    else:
                        success = self.start_3d_visualization_server()
                        if success:
                            print(f"3D visualization enabled at http://localhost:{self.web_server_port}")
                elif key == ord('e'):  # Toggle emotion color
                    self.show_emotion_color_mapping = not self.show_emotion_color_mapping
                    print(f"Emotion color: {'enabled' if self.show_emotion_color_mapping else 'disabled'}")
                elif key == ord('s'):  # Save current frame and data
                    if facial_data:
                        timestamp = int(time.time())
                        cv2.imwrite(f"facial_frame_{timestamp}.jpg", processed_frame)
                        success, filename = self.export_data_to_json(facial_data)
                        if success:
                            print(f"Saved frame and data to facial_frame_{timestamp}.jpg and {filename}")
                        else:
                            print(f"Error saving data: {filename}")
                    else:
                        print("No facial data to save")
                        
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            self.web_server_active = False
            
        except Exception as e:
            print(f"Camera run error: {e}")
            self.web_server_active = False
            cv2.destroyAllWindows()

    def process_image(self, image_path, output_path=None):
        """Process a single image file."""
        try:
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                return False, "Error: Could not read image."
                
            # Process frame
            processed_frame, facial_data = self.process_frame(frame)
            
            if facial_data is None:
                return False, "No face detected in the image."
                
            # Apply enhanced visualization
            processed_frame = self.enhance_visualization(processed_frame, facial_data)
            
            # Save results
            if output_path is None:
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = f"{name}_processed{ext}"
                
            cv2.imwrite(output_path, processed_frame)
            
            # Export facial data
            json_path = os.path.splitext(output_path)[0] + "_data.json"
            success, msg = self.export_data_to_json(facial_data, json_path)
            
            if success:
                return True, f"Results saved to {output_path} and {json_path}"
            else:
                return False, f"Image saved to {output_path}, but data export failed: {msg}"
                
        except Exception as e:
            return False, f"Image processing error: {e}"

    def analyze_video(self, video_path, output_path=None, skip_frames=0):
        """Process a video file."""
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Error: Could not open video."
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup output
            if output_path is None:
                filename = os.path.basename(video_path)
                name, ext = os.path.splitext(filename)
                output_path = f"{name}_processed.mp4"
                
            # Create data output file
            data_path = os.path.splitext(output_path)[0] + "_data.json"
            all_data = {
                'video_info': {
                    'original_path': video_path,
                    'processed_path': output_path,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'total_frames': total_frames,
                    'processed_frames': 0
                },
                'frames': []
            }
            
            # Define codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_idx = 0
            processed_count = 0
            
            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames if requested
                if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                    frame_idx += 1
                    continue
                    
                # Process frame
                processed_frame, facial_data = self.process_frame(frame)
                
                # Write frame
                out.write(processed_frame)
                
                # Save data if face detected
                if facial_data is not None:
                    # Convert landmarks to serializable format
                    landmarks = []
                    for i, lm in enumerate(facial_data['landmarks'].landmark):
                        landmarks.append({
                            'index': i,
                            'x': lm.x,
                            'y': lm.y,
                            'z': lm.z
                        })
                        
                    frame_data = {
                        'frame_idx': frame_idx,
                        'timestamp': frame_idx / fps,
                        'emotion': facial_data['emotion'],
                        'age': facial_data['age'],
                        'age_category': facial_data['age_category'],
                        'head_pose': facial_data['head_pose'],
                        'euler_angles': facial_data['euler_angles'].tolist(),
                        'landmarks': landmarks
                    }
                    
                    all_data['frames'].append(frame_data)
                    processed_count += 1
                    
                # Update progress
                frame_idx += 1
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
                    
            # Clean up
            cap.release()
            out.release()
            
            # Update processed frames count
            all_data['video_info']['processed_frames'] = processed_count
            
            # Save data
            with open(data_path, 'w') as f:
                json.dump(all_data, f, indent=2)
                
            return True, f"Video processed and saved to {output_path}, data saved to {data_path}"
            
        except Exception as e:
            return False, f"Video processing error: {e}"
            
    def batch_process_images(self, folder_path, output_folder=None):
        """Process all images in a folder."""
        try:
            if not os.path.isdir(folder_path):
                return False, "Error: Invalid folder path."
                
            if output_folder is None:
                output_folder = os.path.join(folder_path, "processed")
                
            os.makedirs(output_folder, exist_ok=True)
            
            # Get image files
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = [f for f in os.listdir(folder_path) 
                          if os.path.isfile(os.path.join(folder_path, f)) and 
                          os.path.splitext(f)[1].lower() in extensions]
                          
            if not image_files:
                return False, "No image files found in the folder."
                
            # Process images
            results = []
            for i, img_file in enumerate(image_files):
                input_path = os.path.join(folder_path, img_file)
                output_path = os.path.join(output_folder, f"processed_{img_file}")
                
                print(f"Processing image {i+1}/{len(image_files)}: {img_file}")
                success, msg = self.process_image(input_path, output_path)
                results.append({
                    'file': img_file,
                    'success': success,
                    'message': msg
                })
                
            # Generate summary
            successful = sum(1 for r in results if r['success'])
            
            summary = {
                'total': len(image_files),
                'successful': successful,
                'failed': len(image_files) - successful,
                'output_folder': output_folder,
                'details': results
            }
            
            # Save summary
            summary_path = os.path.join(output_folder, "processing_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
                
            return True, f"Processed {successful}/{len(image_files)} images. Results saved to {output_folder}"
            
        except Exception as e:
            return False, f"Batch processing error: {e}"

# Example usage
if __name__ == "__main__":
    # Create the facial analysis object
    facial_analyzer = FacialAnalysis3D(download_pretrained=True, use_3d=True)
    
    # Run with camera
    facial_analyzer.run_camera(camera_id=0, resolution=(340, 180))
    
    # Process an image
    # success, msg = facial_analyzer.process_image("test_image.jpg")
    # print(msg)
    
    # Process a video
    # success, msg = facial_analyzer.analyze_video("test_video.mp4")
    # print(msg)
    
    # Batch process images
    # success, msg = facial_analyzer.batch_process_images("test_images_folder")
    # print(msg)