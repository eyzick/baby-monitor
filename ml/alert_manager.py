"""
Alert Manager for Baby Monitor
Combines audio and motion ML signals to generate intelligent alerts
while minimizing false positives.
"""

import time
from collections import deque
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import json


class AlertPriority(Enum):
    """Alert priority levels."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertType(Enum):
    """Types of alerts."""
    CRYING = "crying"
    MOTION = "motion"
    NO_MOTION = "no_motion"
    NO_BREATHING = "no_breathing"
    COMBINED = "combined"  # Both crying and motion


@dataclass
class Alert:
    """Represents an alert event."""
    timestamp: float
    alert_type: AlertType
    priority: AlertPriority
    confidence: float
    message: str
    audio_data: Dict = field(default_factory=dict)
    motion_data: Dict = field(default_factory=dict)
    acknowledged: bool = False


class AlertManager:
    """
    Intelligent alert manager that combines audio and motion signals.
    
    Features:
    - Multi-signal fusion for higher accuracy
    - Adaptive cooldown to prevent alert fatigue
    - Alert history tracking
    - Configurable sensitivity profiles
    - False positive suppression
    """
    
    # Sensitivity presets
    SENSITIVITY_PROFILES = {
        "low": {
            "audio_threshold": 0.6,
            "motion_threshold": 0.7,
            "require_confirmation": True,
            "cooldown_seconds": 30,
            "min_duration_seconds": 2.0
        },
        "medium": {
            "audio_threshold": 0.4,
            "motion_threshold": 0.5,
            "require_confirmation": True,
            "cooldown_seconds": 15,
            "min_duration_seconds": 1.0
        },
        "high": {
            "audio_threshold": 0.3,
            "motion_threshold": 0.4,
            "require_confirmation": False,
            "cooldown_seconds": 5,
            "min_duration_seconds": 0.5
        }
    }
    
    def __init__(
        self,
        sensitivity: str = "medium",
        max_history: int = 100,
        no_motion_timeout: float = 60.0,
        callback: Optional[Callable[[Alert], None]] = None
    ):
        """
        Initialize the alert manager.
        
        Args:
            sensitivity: Sensitivity profile ("low", "medium", "high")
            max_history: Maximum number of alerts to keep in history
            no_motion_timeout: Seconds without motion before alerting
            callback: Optional callback function for new alerts
        """
        self.sensitivity = sensitivity
        self.profile = self.SENSITIVITY_PROFILES.get(sensitivity, self.SENSITIVITY_PROFILES["medium"])
        self.max_history = max_history
        self.no_motion_timeout = no_motion_timeout
        self.callback = callback
        
        # Alert state
        self.current_alert: Optional[Alert] = None
        self.alert_history: deque = deque(maxlen=max_history)
        self.last_alert_time = 0
        self.alert_start_time = 0
        self.pending_alert = False
        
        # Signal tracking for confirmation
        self.audio_signals = deque(maxlen=10)
        self.motion_signals = deque(maxlen=10)
        
        # Statistics
        self.stats = {
            "total_alerts": 0,
            "crying_alerts": 0,
            "motion_alerts": 0,
            "false_positives": 0,
            "suppressed_alerts": 0
        }
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        print(f"Alert Manager initialized with '{sensitivity}' sensitivity")
        print(f"  - Audio threshold: {self.profile['audio_threshold']}")
        print(f"  - Motion threshold: {self.profile['motion_threshold']}")
    
    def set_sensitivity(self, sensitivity: str):
        """Change sensitivity profile."""
        if sensitivity in self.SENSITIVITY_PROFILES:
            with self._lock:
                self.sensitivity = sensitivity
                self.profile = self.SENSITIVITY_PROFILES[sensitivity]
                print(f"Sensitivity changed to '{sensitivity}'")
    
    def _check_cooldown(self) -> bool:
        """Check if cooldown period has passed."""
        return (time.time() - self.last_alert_time) > self.profile["cooldown_seconds"]
    
    def _calculate_priority(
        self,
        audio_confidence: float,
        motion_confidence: float,
        alert_type: AlertType
    ) -> AlertPriority:
        """Calculate alert priority based on confidence and type."""
        combined = (audio_confidence + motion_confidence) / 2
        
        # No breathing is always critical
        if alert_type == AlertType.NO_BREATHING:
            return AlertPriority.CRITICAL
        
        # Combined crying + motion is higher priority
        if alert_type == AlertType.COMBINED:
            if combined > 0.7:
                return AlertPriority.HIGH
            elif combined > 0.4:
                return AlertPriority.MEDIUM
            return AlertPriority.LOW
        
        # Crying alerts
        if alert_type == AlertType.CRYING:
            if audio_confidence > 0.7:
                return AlertPriority.HIGH
            elif audio_confidence > 0.5:
                return AlertPriority.MEDIUM
            return AlertPriority.LOW
        
        # Motion alerts
        if alert_type == AlertType.MOTION:
            if motion_confidence > 0.8:
                return AlertPriority.MEDIUM
            return AlertPriority.LOW
        
        return AlertPriority.NONE
    
    def _generate_message(self, alert_type: AlertType, audio_data: Dict, motion_data: Dict) -> str:
        """Generate human-readable alert message."""
        messages = {
            AlertType.CRYING: f"Baby crying detected ({audio_data.get('detected_class', 'cry')})",
            AlertType.MOTION: f"Significant movement detected",
            AlertType.COMBINED: f"Baby is crying and moving",
            AlertType.NO_MOTION: f"No movement detected for extended period",
            AlertType.NO_BREATHING: f"⚠️ No breathing pattern detected - please check baby"
        }
        
        base_msg = messages.get(alert_type, "Alert")
        confidence = max(audio_data.get("confidence", 0), motion_data.get("confidence", 0))
        
        return f"{base_msg} (confidence: {confidence:.0%})"
    
    def _confirm_signal(self, signals: deque, threshold: float) -> bool:
        """
        Confirm signal by checking recent history.
        
        Requires multiple detections above threshold to confirm.
        """
        if len(signals) < 3:
            return False
        
        recent = list(signals)[-5:]
        above_threshold = sum(1 for s in recent if s > threshold)
        
        return above_threshold >= 3
    
    def process(self, audio_result: Dict, motion_result: Dict) -> Optional[Alert]:
        """
        Process audio and motion results to determine if an alert should be raised.
        
        Args:
            audio_result: Result from AudioClassifier.classify()
            motion_result: Result from MotionDetector.detect()
        
        Returns:
            Alert object if alert should be raised, None otherwise
        """
        with self._lock:
            current_time = time.time()
            
            # Extract signals
            audio_confidence = audio_result.get("confidence", 0.0)
            motion_confidence = motion_result.get("confidence", 0.0)
            is_crying = audio_result.get("is_crying", False)
            is_motion = motion_result.get("motion_detected", False)
            breathing = motion_result.get("breathing_detected", False)
            no_motion_alert = motion_result.get("no_motion_alert", False)
            
            # Track signals for confirmation
            self.audio_signals.append(audio_confidence if is_crying else 0)
            self.motion_signals.append(motion_confidence if is_motion else 0)
            
            # Check for no-breathing alert (highest priority)
            if no_motion_alert and not breathing:
                alert = Alert(
                    timestamp=current_time,
                    alert_type=AlertType.NO_BREATHING,
                    priority=AlertPriority.CRITICAL,
                    confidence=1.0,
                    message=self._generate_message(AlertType.NO_BREATHING, audio_result, motion_result),
                    audio_data=audio_result,
                    motion_data=motion_result
                )
                self._emit_alert(alert)
                return alert
            
            # Check cooldown
            if not self._check_cooldown():
                return None
            
            # Determine alert type
            alert_type = None
            
            # Combined alert (both crying and motion)
            if is_crying and is_motion:
                if audio_confidence > self.profile["audio_threshold"] * 0.8 and \
                   motion_confidence > self.profile["motion_threshold"] * 0.8:
                    alert_type = AlertType.COMBINED
            
            # Crying alert
            elif is_crying and audio_confidence > self.profile["audio_threshold"]:
                if self.profile["require_confirmation"]:
                    if self._confirm_signal(self.audio_signals, self.profile["audio_threshold"]):
                        alert_type = AlertType.CRYING
                else:
                    alert_type = AlertType.CRYING
            
            # Motion alert
            elif is_motion and motion_confidence > self.profile["motion_threshold"]:
                motion_type = motion_result.get("motion_type", "")
                # Only alert for significant or high activity motion
                if motion_type in ["significant", "high"]:
                    if self.profile["require_confirmation"]:
                        if self._confirm_signal(self.motion_signals, self.profile["motion_threshold"]):
                            alert_type = AlertType.MOTION
                    else:
                        alert_type = AlertType.MOTION
            
            # No alert needed
            if alert_type is None:
                self.pending_alert = False
                return None
            
            # Check minimum duration for pending alerts
            if not self.pending_alert:
                self.pending_alert = True
                self.alert_start_time = current_time
                return None
            
            elapsed = current_time - self.alert_start_time
            if elapsed < self.profile["min_duration_seconds"]:
                return None
            
            # Generate alert
            priority = self._calculate_priority(audio_confidence, motion_confidence, alert_type)
            
            alert = Alert(
                timestamp=current_time,
                alert_type=alert_type,
                priority=priority,
                confidence=max(audio_confidence, motion_confidence),
                message=self._generate_message(alert_type, audio_result, motion_result),
                audio_data=audio_result.copy(),
                motion_data=motion_result.copy()
            )
            
            self._emit_alert(alert)
            return alert
    
    def _emit_alert(self, alert: Alert):
        """Emit an alert (update state, history, and trigger callback)."""
        self.current_alert = alert
        self.alert_history.append(alert)
        self.last_alert_time = alert.timestamp
        self.pending_alert = False
        
        # Update stats
        self.stats["total_alerts"] += 1
        if alert.alert_type == AlertType.CRYING:
            self.stats["crying_alerts"] += 1
        elif alert.alert_type == AlertType.MOTION:
            self.stats["motion_alerts"] += 1
        
        # Trigger callback
        if self.callback:
            try:
                self.callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
    
    def acknowledge_alert(self):
        """Acknowledge the current alert."""
        with self._lock:
            if self.current_alert:
                self.current_alert.acknowledged = True
    
    def report_false_positive(self):
        """Report the last alert as a false positive (for learning)."""
        with self._lock:
            self.stats["false_positives"] += 1
            # Could be used for online learning in the future
    
    def get_current_alert(self) -> Optional[Dict]:
        """Get current alert as dictionary for API."""
        with self._lock:
            if self.current_alert is None:
                return None
            
            # Clear alert if it's old (more than 10 seconds)
            if time.time() - self.current_alert.timestamp > 10:
                self.current_alert = None
                return None
            
            return {
                "timestamp": self.current_alert.timestamp,
                "type": self.current_alert.alert_type.value,
                "priority": self.current_alert.priority.name.lower(),
                "confidence": round(self.current_alert.confidence, 3),
                "message": self.current_alert.message,
                "acknowledged": self.current_alert.acknowledged
            }
    
    def get_status(self) -> Dict:
        """Get alert manager status for API."""
        with self._lock:
            current = self.get_current_alert()
            
            return {
                "has_alert": current is not None,
                "current_alert": current,
                "sensitivity": self.sensitivity,
                "cooldown_remaining": max(0, self.profile["cooldown_seconds"] - (time.time() - self.last_alert_time)),
                "stats": self.stats.copy()
            }
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent alert history."""
        with self._lock:
            alerts = list(self.alert_history)[-limit:]
            return [
                {
                    "timestamp": a.timestamp,
                    "type": a.alert_type.value,
                    "priority": a.priority.name.lower(),
                    "confidence": round(a.confidence, 3),
                    "message": a.message
                }
                for a in reversed(alerts)
            ]
    
    def reset(self):
        """Reset alert manager state."""
        with self._lock:
            self.current_alert = None
            self.pending_alert = False
            self.audio_signals.clear()
            self.motion_signals.clear()
            self.last_alert_time = 0



