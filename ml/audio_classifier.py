"""
Audio Classifier for Baby Monitor
Uses YAMNet (TensorFlow Lite) to detect infant cries and distinguish from background noise.
Optimized for low false alerts while maintaining sensitivity.
"""

import numpy as np
import time
from collections import deque
from typing import Dict, Tuple, Optional
import threading

# TensorFlow Lite for efficient inference on Raspberry Pi
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

import urllib.request
import os


class AudioClassifier:
    """
    ML-powered audio classifier to detect infant cries.
    
    Uses YAMNet, a pre-trained audio event classifier that recognizes 521 audio classes
    including "Baby cry, infant cry" with high accuracy.
    
    Features:
    - Temporal smoothing to reduce false positives
    - Adaptive thresholding based on ambient noise levels
    - Confidence scoring with multiple detection tiers
    """
    
    # YAMNet class indices for baby-related sounds
    BABY_CRY_CLASSES = {
        20: "Baby cry, infant cry",
        21: "Baby laughter",
        22: "Whimper",
    }
    
    # Additional classes that might indicate baby activity
    ALERT_CLASSES = {
        23: "Crying, sobbing",
        24: "Sigh",
        25: "Screaming",
        394: "Gasp",
    }
    
    # Classes to ignore (common background noises)
    IGNORE_CLASSES = {
        0: "Speech",
        1: "Child speech, kid speaking",
        137: "Music",
        494: "Silence",
        495: "White noise",
    }
    
    MODEL_URL = "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "yamnet.tflite")
    LABELS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    
    def __init__(
        self,
        sample_rate: int = 16000,
        cry_threshold: float = 0.35,
        alert_threshold: float = 0.50,
        smoothing_window: int = 5,
        cooldown_seconds: float = 2.0,
        adaptive_threshold: bool = True
    ):
        """
        Initialize the audio classifier.
        
        Args:
            sample_rate: Audio sample rate (YAMNet expects 16kHz)
            cry_threshold: Minimum confidence to trigger cry detection (lower = more sensitive)
            alert_threshold: Confidence for high-priority alerts
            smoothing_window: Number of frames to smooth predictions over
            cooldown_seconds: Minimum time between alerts to reduce spam
            adaptive_threshold: Enable adaptive thresholding based on ambient noise
        """
        self.sample_rate = sample_rate
        self.cry_threshold = cry_threshold
        self.alert_threshold = alert_threshold
        self.smoothing_window = smoothing_window
        self.cooldown_seconds = cooldown_seconds
        self.adaptive_threshold = adaptive_threshold
        
        # State tracking
        self.prediction_history = deque(maxlen=smoothing_window)
        self.ambient_noise_level = 0.0
        self.last_alert_time = 0
        self.is_crying = False
        self.confidence = 0.0
        self.detected_class = ""
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Load model
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model()
    
    def _download_model(self):
        """Download the YAMNet TFLite model if not present."""
        if not os.path.exists(self.MODEL_PATH):
            print("Downloading YAMNet model for audio classification...")
            # Use a direct TFLite model URL
            model_url = "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/yamnet/classification/tflite/1.tflite"
            try:
                urllib.request.urlretrieve(model_url, self.MODEL_PATH)
                print("YAMNet model downloaded successfully.")
            except Exception as e:
                print(f"Failed to download model: {e}")
                print("Audio ML classification will be disabled.")
                return False
        return True
    
    def _load_model(self):
        """Load the TFLite model for inference."""
        try:
            if not self._download_model():
                return
            
            self.interpreter = tflite.Interpreter(model_path=self.MODEL_PATH)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print("YAMNet audio classifier loaded successfully.")
            print(f"  - Cry detection threshold: {self.cry_threshold}")
            print(f"  - Alert threshold: {self.alert_threshold}")
            
        except Exception as e:
            print(f"Failed to load audio classifier model: {e}")
            self.interpreter = None
    
    def _preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for YAMNet.
        
        YAMNet expects:
        - Mono audio
        - 16kHz sample rate
        - Float32 normalized to [-1, 1]
        - 0.975 second windows (15600 samples at 16kHz)
        """
        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Ensure mono
        if len(audio_float.shape) > 1:
            audio_float = np.mean(audio_float, axis=1)
        
        return audio_float
    
    def _apply_temporal_smoothing(self, raw_scores: Dict[int, float]) -> Dict[int, float]:
        """
        Apply temporal smoothing to reduce false positives from transient sounds.
        
        Uses a sliding window average of recent predictions.
        """
        self.prediction_history.append(raw_scores)
        
        if len(self.prediction_history) < 2:
            return raw_scores
        
        # Average scores across the window
        smoothed = {}
        all_classes = set()
        for scores in self.prediction_history:
            all_classes.update(scores.keys())
        
        for class_id in all_classes:
            values = [scores.get(class_id, 0.0) for scores in self.prediction_history]
            smoothed[class_id] = np.mean(values)
        
        return smoothed
    
    def _update_ambient_noise(self, audio_data: np.ndarray):
        """Update the ambient noise level estimate using exponential moving average."""
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        alpha = 0.1  # Smoothing factor
        self.ambient_noise_level = alpha * rms + (1 - alpha) * self.ambient_noise_level
    
    def _get_adaptive_threshold(self) -> float:
        """
        Calculate adaptive threshold based on ambient noise level.
        
        In noisy environments, we raise the threshold to reduce false positives.
        In quiet environments, we can be more sensitive.
        """
        if not self.adaptive_threshold:
            return self.cry_threshold
        
        # Scale threshold based on ambient noise (0.0 to 0.3 typical range)
        noise_factor = min(self.ambient_noise_level * 2, 0.2)
        return self.cry_threshold + noise_factor
    
    def classify(self, audio_data: np.ndarray) -> Dict:
        """
        Classify audio data and detect infant cries.
        
        Args:
            audio_data: Raw audio samples (int16 or float32)
        
        Returns:
            Dictionary with:
                - is_crying: bool - Whether an infant cry was detected
                - confidence: float - Confidence level (0-1)
                - detected_class: str - Name of detected sound class
                - alert_level: str - "none", "low", "medium", "high"
                - raw_scores: dict - Raw class scores for debugging
        """
        result = {
            "is_crying": False,
            "confidence": 0.0,
            "detected_class": "",
            "alert_level": "none",
            "raw_scores": {}
        }
        
        if self.interpreter is None or len(audio_data) < 1000:
            return result
        
        with self._lock:
            try:
                # Update ambient noise estimate
                self._update_ambient_noise(audio_data)
                
                # Preprocess audio
                audio_float = self._preprocess_audio(audio_data)
                
                # Ensure correct length for YAMNet (pad or trim to ~1 second)
                target_length = self.sample_rate  # 1 second
                if len(audio_float) < target_length:
                    audio_float = np.pad(audio_float, (0, target_length - len(audio_float)))
                else:
                    audio_float = audio_float[:target_length]
                
                # Run inference
                self.interpreter.set_tensor(
                    self.input_details[0]['index'],
                    audio_float.astype(np.float32)
                )
                self.interpreter.invoke()
                
                # Get scores (YAMNet outputs scores for 521 classes)
                scores = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                # Handle multi-frame output (YAMNet may output multiple time frames)
                if len(scores.shape) > 1:
                    scores = np.mean(scores, axis=0)
                
                # Extract relevant class scores
                raw_scores = {}
                for class_id in list(self.BABY_CRY_CLASSES.keys()) + list(self.ALERT_CLASSES.keys()):
                    if class_id < len(scores):
                        raw_scores[class_id] = float(scores[class_id])
                
                result["raw_scores"] = raw_scores
                
                # Apply temporal smoothing
                smoothed_scores = self._apply_temporal_smoothing(raw_scores)
                
                # Get adaptive threshold
                threshold = self._get_adaptive_threshold()
                
                # Check for baby cry (highest priority)
                max_cry_score = 0.0
                max_cry_class = ""
                
                for class_id, class_name in self.BABY_CRY_CLASSES.items():
                    score = smoothed_scores.get(class_id, 0.0)
                    if score > max_cry_score:
                        max_cry_score = score
                        max_cry_class = class_name
                
                # Check for other alert sounds
                max_alert_score = 0.0
                max_alert_class = ""
                
                for class_id, class_name in self.ALERT_CLASSES.items():
                    score = smoothed_scores.get(class_id, 0.0)
                    if score > max_alert_score:
                        max_alert_score = score
                        max_alert_class = class_name
                
                # Determine detection result
                current_time = time.time()
                cooldown_passed = (current_time - self.last_alert_time) > self.cooldown_seconds
                
                if max_cry_score > threshold and cooldown_passed:
                    result["is_crying"] = True
                    result["confidence"] = max_cry_score
                    result["detected_class"] = max_cry_class
                    
                    if max_cry_score > self.alert_threshold:
                        result["alert_level"] = "high"
                    elif max_cry_score > threshold + 0.1:
                        result["alert_level"] = "medium"
                    else:
                        result["alert_level"] = "low"
                    
                    self.last_alert_time = current_time
                    
                elif max_alert_score > threshold + 0.1 and cooldown_passed:
                    # Secondary alert for other concerning sounds
                    result["is_crying"] = True
                    result["confidence"] = max_alert_score
                    result["detected_class"] = max_alert_class
                    result["alert_level"] = "low"
                    self.last_alert_time = current_time
                
                # Update state
                self.is_crying = result["is_crying"]
                self.confidence = result["confidence"]
                self.detected_class = result["detected_class"]
                
            except Exception as e:
                print(f"Audio classification error: {e}")
        
        return result
    
    def get_status(self) -> Dict:
        """Get current classifier status for API endpoints."""
        with self._lock:
            return {
                "is_crying": self.is_crying,
                "confidence": round(self.confidence, 3),
                "detected_class": self.detected_class,
                "ambient_noise": round(self.ambient_noise_level, 4),
                "threshold": round(self._get_adaptive_threshold(), 3),
                "model_loaded": self.interpreter is not None
            }
    
    def reset(self):
        """Reset classifier state."""
        with self._lock:
            self.prediction_history.clear()
            self.is_crying = False
            self.confidence = 0.0
            self.detected_class = ""
            self.last_alert_time = 0



