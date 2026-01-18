"""
ML-Enhanced Motion Detector for Baby Monitor
Uses advanced computer vision and statistical analysis to detect meaningful
motion events while filtering out false positives from shadows, lighting changes, etc.
"""

import numpy as np
import cv2
import time
from collections import deque
from typing import Dict, Tuple, Optional, List
import threading
from dataclasses import dataclass
from enum import Enum


class MotionType(Enum):
    """Classification of motion types."""
    NONE = "none"
    NOISE = "noise"  # Camera noise, compression artifacts
    LIGHTING = "lighting"  # Lighting changes, shadows
    SMALL_MOVEMENT = "small"  # Minor movements (breathing, small shifts)
    SIGNIFICANT_MOVEMENT = "significant"  # Larger movements (rolling, sitting up)
    HIGH_ACTIVITY = "high"  # Very active (standing, crawling)


@dataclass
class MotionEvent:
    """Represents a detected motion event."""
    timestamp: float
    motion_type: MotionType
    confidence: float
    area_ratio: float
    centroid: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h


class MotionDetector:
    """
    ML-enhanced motion detector that distinguishes meaningful baby movements
    from background noise, shadows, and lighting changes.
    
    Features:
    - Adaptive background subtraction
    - Multi-frame temporal analysis
    - Motion pattern classification
    - Region of Interest (ROI) support for crib area
    - Breathing detection mode
    """
    
    def __init__(
        self,
        motion_threshold: float = 0.4,
        min_area_ratio: float = 0.001,
        max_area_ratio: float = 0.5,
        history_frames: int = 30,
        learning_rate: float = 0.01,
        enable_breathing_detection: bool = True,
        roi: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Initialize the motion detector.
        
        Args:
            motion_threshold: Confidence threshold for motion alerts (0-1)
            min_area_ratio: Minimum motion area as ratio of frame (filters noise)
            max_area_ratio: Maximum motion area as ratio of frame (filters lighting changes)
            history_frames: Number of frames for background model
            learning_rate: Background model learning rate (lower = more stable)
            enable_breathing_detection: Track subtle periodic motion
            roi: Region of interest (x, y, width, height) - None for full frame
        """
        self.motion_threshold = motion_threshold
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.history_frames = history_frames
        self.learning_rate = learning_rate
        self.enable_breathing_detection = enable_breathing_detection
        self.roi = roi
        
        # Background subtractor with shadow detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history_frames,
            varThreshold=50,
            detectShadows=True
        )
        
        # Alternative: KNN background subtractor (better for dynamic backgrounds)
        self.bg_subtractor_knn = cv2.createBackgroundSubtractorKNN(
            history=history_frames,
            dist2Threshold=400,
            detectShadows=True
        )
        
        # Motion history
        self.motion_history = deque(maxlen=60)  # 2 seconds at 30fps
        self.frame_history = deque(maxlen=10)
        self.breathing_history = deque(maxlen=300)  # 10 seconds for breathing pattern
        
        # State
        self.motion_detected = False
        self.motion_type = MotionType.NONE
        self.confidence = 0.0
        self.last_motion_time = 0
        self.frame_count = 0
        self.frame_size = (640, 480)
        
        # Breathing detection state
        self.breathing_detected = False
        self.breathing_rate = 0.0
        self.no_motion_duration = 0.0
        self.last_frame_time = time.time()
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        print("ML Motion Detector initialized")
        print(f"  - Motion threshold: {motion_threshold}")
        print(f"  - Breathing detection: {enable_breathing_detection}")
    
    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract region of interest from frame."""
        if self.roi is None:
            return frame
        
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w]
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for motion analysis."""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        return blurred
    
    def _analyze_motion_pattern(
        self,
        contours: List,
        frame_area: int
    ) -> Tuple[MotionType, float, List[MotionEvent]]:
        """
        Analyze motion contours to classify the type of motion.
        
        Uses multiple heuristics:
        - Total motion area
        - Number of distinct regions
        - Shape characteristics
        - Temporal consistency
        """
        if not contours:
            return MotionType.NONE, 0.0, []
        
        events = []
        total_area = 0
        significant_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / frame_area
            
            # Filter by area
            if area_ratio < self.min_area_ratio:
                continue
            
            total_area += area
            significant_contours.append(contour)
            
            # Get contour properties
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(contour)
            
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            events.append(MotionEvent(
                timestamp=time.time(),
                motion_type=MotionType.SMALL_MOVEMENT,
                confidence=min(area_ratio * 10, 1.0),
                area_ratio=area_ratio,
                centroid=(cx, cy),
                bounding_box=(x, y, w, h)
            ))
        
        if not significant_contours:
            return MotionType.NONE, 0.0, []
        
        total_area_ratio = total_area / frame_area
        num_regions = len(significant_contours)
        
        # Classification logic
        confidence = 0.0
        motion_type = MotionType.NONE
        
        # Check for lighting change (large area, uniform distribution)
        if total_area_ratio > self.max_area_ratio:
            motion_type = MotionType.LIGHTING
            confidence = 0.2  # Low confidence for lighting changes
        
        # Check for camera noise (many tiny regions)
        elif num_regions > 20 and total_area_ratio < 0.05:
            motion_type = MotionType.NOISE
            confidence = 0.1
        
        # Significant movement
        elif total_area_ratio > 0.02 and num_regions < 10:
            motion_type = MotionType.SIGNIFICANT_MOVEMENT
            confidence = min(total_area_ratio * 5, 0.95)
        
        # High activity
        elif total_area_ratio > 0.1:
            motion_type = MotionType.HIGH_ACTIVITY
            confidence = min(total_area_ratio * 3, 0.99)
        
        # Small movement
        elif total_area_ratio > self.min_area_ratio:
            motion_type = MotionType.SMALL_MOVEMENT
            confidence = min(total_area_ratio * 10, 0.7)
        
        # Update events with classified type
        for event in events:
            event.motion_type = motion_type
        
        return motion_type, confidence, events
    
    def _apply_temporal_smoothing(
        self,
        motion_type: MotionType,
        confidence: float
    ) -> Tuple[MotionType, float]:
        """
        Apply temporal smoothing to reduce false positives from transient motion.
        
        Requires consistent motion across multiple frames to trigger alert.
        """
        self.motion_history.append((motion_type, confidence))
        
        if len(self.motion_history) < 3:
            return motion_type, confidence
        
        # Get recent history
        recent = list(self.motion_history)[-5:]
        
        # Count motion types
        type_counts = {}
        total_confidence = 0
        
        for mt, conf in recent:
            type_counts[mt] = type_counts.get(mt, 0) + 1
            if mt != MotionType.NONE and mt != MotionType.NOISE:
                total_confidence += conf
        
        # Require at least 3/5 frames with motion for detection
        significant_types = [MotionType.SMALL_MOVEMENT, MotionType.SIGNIFICANT_MOVEMENT, MotionType.HIGH_ACTIVITY]
        significant_count = sum(type_counts.get(t, 0) for t in significant_types)
        
        if significant_count >= 3:
            # Return the most common significant type
            best_type = MotionType.NONE
            best_count = 0
            for t in significant_types:
                if type_counts.get(t, 0) > best_count:
                    best_count = type_counts.get(t, 0)
                    best_type = t
            
            smoothed_confidence = total_confidence / len(recent)
            return best_type, smoothed_confidence
        
        return MotionType.NONE, 0.0
    
    def _analyze_breathing(self, motion_values: List[float]) -> Tuple[bool, float]:
        """
        Analyze subtle periodic motion patterns to detect breathing.
        
        Uses FFT to find periodic signals in the 0.15-0.5 Hz range
        (9-30 breaths per minute, typical for infants).
        """
        if len(motion_values) < 60:  # Need at least 2 seconds of data
            return False, 0.0
        
        try:
            # Apply FFT
            values = np.array(motion_values)
            values = values - np.mean(values)  # Remove DC component
            
            fft = np.abs(np.fft.rfft(values))
            freqs = np.fft.rfftfreq(len(values), d=1/30)  # Assuming 30 fps
            
            # Look for peaks in breathing range (0.15-0.5 Hz = 9-30 bpm)
            breathing_mask = (freqs >= 0.15) & (freqs <= 0.5)
            breathing_fft = fft[breathing_mask]
            breathing_freqs = freqs[breathing_mask]
            
            if len(breathing_fft) == 0:
                return False, 0.0
            
            # Find dominant frequency
            peak_idx = np.argmax(breathing_fft)
            peak_freq = breathing_freqs[peak_idx]
            peak_power = breathing_fft[peak_idx]
            
            # Calculate signal-to-noise ratio
            noise_power = np.mean(fft[~breathing_mask]) if np.any(~breathing_mask) else 1.0
            snr = peak_power / (noise_power + 1e-6)
            
            # Breathing detected if SNR is above threshold
            if snr > 3.0 and peak_power > np.std(values) * 2:
                breathing_rate = peak_freq * 60  # Convert to breaths per minute
                return True, breathing_rate
            
        except Exception as e:
            print(f"Breathing analysis error: {e}")
        
        return False, 0.0
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Analyze a frame for motion and return detection results.
        
        Args:
            frame: Input frame (BGR or grayscale)
        
        Returns:
            Dictionary with detection results
        """
        result = {
            "motion_detected": False,
            "motion_type": "none",
            "confidence": 0.0,
            "alert_level": "none",
            "breathing_detected": False,
            "breathing_rate": 0.0,
            "events": [],
            "no_motion_alert": False
        }
        
        with self._lock:
            try:
                current_time = time.time()
                dt = current_time - self.last_frame_time
                self.last_frame_time = current_time
                self.frame_count += 1
                
                # Update frame size
                self.frame_size = (frame.shape[1], frame.shape[0])
                frame_area = frame.shape[0] * frame.shape[1]
                
                # Extract ROI if set
                roi_frame = self._extract_roi(frame)
                roi_area = roi_frame.shape[0] * roi_frame.shape[1]
                
                # Preprocess
                processed = self._preprocess_frame(roi_frame)
                
                # Apply background subtraction
                fg_mask = self.bg_subtractor.apply(processed, learningRate=self.learning_rate)
                
                # Remove shadows (marked as 127 in MOG2)
                fg_mask[fg_mask == 127] = 0
                
                # Morphological operations to clean up mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
                
                # Find contours
                contours, _ = cv2.findContours(
                    fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Analyze motion patterns
                raw_motion_type, raw_confidence, events = self._analyze_motion_pattern(
                    contours, roi_area
                )
                
                # Apply temporal smoothing
                motion_type, confidence = self._apply_temporal_smoothing(
                    raw_motion_type, raw_confidence
                )
                
                # Store motion intensity for breathing detection
                motion_intensity = np.sum(fg_mask > 0) / roi_area
                self.breathing_history.append(motion_intensity)
                
                # Determine if motion should trigger alert
                is_significant = motion_type in [
                    MotionType.SIGNIFICANT_MOVEMENT,
                    MotionType.HIGH_ACTIVITY
                ]
                
                if is_significant and confidence > self.motion_threshold:
                    result["motion_detected"] = True
                    result["motion_type"] = motion_type.value
                    result["confidence"] = confidence
                    
                    if motion_type == MotionType.HIGH_ACTIVITY:
                        result["alert_level"] = "high"
                    elif confidence > 0.7:
                        result["alert_level"] = "medium"
                    else:
                        result["alert_level"] = "low"
                    
                    self.motion_detected = True
                    self.last_motion_time = current_time
                    self.no_motion_duration = 0.0
                    
                elif motion_type == MotionType.SMALL_MOVEMENT:
                    # Small movement - track but don't alert
                    self.no_motion_duration = 0.0
                    result["motion_type"] = motion_type.value
                    result["confidence"] = confidence
                    
                else:
                    # No significant motion
                    self.no_motion_duration += dt
                    self.motion_detected = False
                
                # Breathing detection
                if self.enable_breathing_detection:
                    breathing_detected, breathing_rate = self._analyze_breathing(
                        list(self.breathing_history)
                    )
                    result["breathing_detected"] = breathing_detected
                    result["breathing_rate"] = round(breathing_rate, 1)
                    self.breathing_detected = breathing_detected
                    self.breathing_rate = breathing_rate
                
                # No motion alert (baby hasn't moved in a while)
                # Only alert if we also don't detect breathing
                if self.no_motion_duration > 30.0 and not self.breathing_detected:
                    result["no_motion_alert"] = True
                
                # Store state
                self.motion_type = motion_type
                self.confidence = confidence
                result["events"] = [
                    {
                        "type": e.motion_type.value,
                        "confidence": e.confidence,
                        "area": e.area_ratio,
                        "position": e.centroid
                    }
                    for e in events[:5]  # Limit to 5 events
                ]
                
            except Exception as e:
                print(f"Motion detection error: {e}")
        
        return result
    
    def get_status(self) -> Dict:
        """Get current detector status for API endpoints."""
        with self._lock:
            return {
                "motion_detected": self.motion_detected,
                "motion_type": self.motion_type.value,
                "confidence": round(self.confidence, 3),
                "breathing_detected": self.breathing_detected,
                "breathing_rate": round(self.breathing_rate, 1),
                "no_motion_seconds": round(self.no_motion_duration, 1),
                "frame_count": self.frame_count
            }
    
    def set_roi(self, x: int, y: int, width: int, height: int):
        """Set region of interest for motion detection."""
        with self._lock:
            self.roi = (x, y, width, height)
            # Reset background model for new ROI
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history_frames,
                varThreshold=50,
                detectShadows=True
            )
    
    def clear_roi(self):
        """Clear region of interest (use full frame)."""
        with self._lock:
            self.roi = None
    
    def reset(self):
        """Reset detector state."""
        with self._lock:
            self.motion_history.clear()
            self.breathing_history.clear()
            self.motion_detected = False
            self.motion_type = MotionType.NONE
            self.confidence = 0.0
            self.no_motion_duration = 0.0
            self.breathing_detected = False
            self.breathing_rate = 0.0
            
            # Reset background model
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=self.history_frames,
                varThreshold=50,
                detectShadows=True
            )



