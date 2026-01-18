from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import numpy as np
import time
import pyaudio
import struct
import scipy.signal

# Import ML modules
from ml import AudioClassifier, MotionDetector, AlertManager

app = Flask(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Video Settings
camera = cv2.VideoCapture(0)

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 4096

# ML Audio requires 16kHz, so we'll resample
ML_SAMPLE_RATE = 16000

# =============================================================================
# ML COMPONENTS
# =============================================================================

# Initialize ML classifiers
audio_classifier = AudioClassifier(
    sample_rate=ML_SAMPLE_RATE,
    cry_threshold=0.35,
    alert_threshold=0.50,
    smoothing_window=5,
    cooldown_seconds=2.0,
    adaptive_threshold=True
)

motion_detector = MotionDetector(
    motion_threshold=0.4,
    min_area_ratio=0.001,
    max_area_ratio=0.5,
    history_frames=30,
    learning_rate=0.01,
    enable_breathing_detection=True
)

alert_manager = AlertManager(
    sensitivity="medium",
    max_history=100,
    no_motion_timeout=60.0
)

# =============================================================================
# STATE VARIABLES
# =============================================================================

# Audio ML state
audio_ml_result = {
    "is_crying": False,
    "confidence": 0.0,
    "detected_class": "",
    "alert_level": "none"
}
audio_ml_lock = threading.Lock()

# Motion ML state
motion_ml_result = {
    "motion_detected": False,
    "motion_type": "none",
    "confidence": 0.0,
    "breathing_detected": False,
    "breathing_rate": 0.0
}
motion_ml_lock = threading.Lock()

# Legacy compatibility
motion_detected = False
last_motion_time = 0

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def resample_audio(audio_data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio from original rate to target rate."""
    if orig_rate == target_rate:
        return audio_data
    
    # Calculate number of samples in resampled audio
    num_samples = int(len(audio_data) * target_rate / orig_rate)
    
    # Use scipy for high-quality resampling
    resampled = scipy.signal.resample(audio_data, num_samples)
    
    return resampled.astype(np.int16)


def get_error_frame():
    """Generates a placeholder image when the camera is unavailable."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    cv2.putText(frame, "NO SIGNAL", (180, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(frame, f"Time: {timestamp}", (200, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

# =============================================================================
# VIDEO STREAMING WITH ML MOTION DETECTION
# =============================================================================

def generate_frames():
    global motion_detected, last_motion_time, motion_ml_result
    
    while True:
        success, frame = camera.read()
        
        if not success:
            frame_data = get_error_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(1)
            
            if not camera.isOpened():
                camera.open(0)
            continue

        # ML Motion Detection
        result = motion_detector.detect(frame)
        
        with motion_ml_lock:
            motion_ml_result = result
            motion_detected = result.get("motion_detected", False)
            if motion_detected:
                last_motion_time = time.time()
        
        # Process alerts (combine with latest audio result)
        with audio_ml_lock:
            current_audio = audio_ml_result.copy()
        alert_manager.process(current_audio, result)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_data = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

# =============================================================================
# AUDIO STREAMING WITH ML CRY DETECTION
# =============================================================================

def gen_audio():
    global audio_ml_result
    
    p = pyaudio.PyAudio()
    
    # Device selection logic
    print("\n--- Audio Setup ---")
    device_count = p.get_device_count()
    input_device_index = None
    
    for i in range(device_count):
        try:
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                if input_device_index is None:
                    input_device_index = i
        except:
            continue

    if input_device_index is None:
        while True:
            yield b'\x00' * CHUNK
            time.sleep(CHUNK / RATE)
        return

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=input_device_index,
                        frames_per_buffer=CHUNK)
    except OSError as e:
        print(f"Audio Open Error: {e}")
        return

    # Buffer for ML analysis (collect ~1 second of audio)
    ml_buffer = []
    ml_buffer_samples = 0
    target_samples = RATE  # 1 second of audio at original rate

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert bytes to numpy array of int16
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Add to ML buffer
            ml_buffer.append(audio_data)
            ml_buffer_samples += len(audio_data)
            
            # When we have enough audio, run ML classification
            if ml_buffer_samples >= target_samples:
                # Concatenate buffer
                full_audio = np.concatenate(ml_buffer)
                
                # Resample to 16kHz for ML model
                resampled = resample_audio(full_audio, RATE, ML_SAMPLE_RATE)
                
                # Run ML classification
                result = audio_classifier.classify(resampled)
                
                with audio_ml_lock:
                    audio_ml_result = result
                
                # Clear buffer (keep some overlap for continuity)
                overlap_samples = RATE // 4  # 250ms overlap
                if ml_buffer_samples > overlap_samples:
                    # Keep last portion
                    keep_audio = full_audio[-overlap_samples:]
                    ml_buffer = [keep_audio]
                    ml_buffer_samples = len(keep_audio)
                else:
                    ml_buffer = []
                    ml_buffer_samples = 0
            
            # Intelligent noise gate based on ML results
            with audio_ml_lock:
                is_crying = audio_ml_result.get("is_crying", False)
            
            # Calculate peak amplitude
            peak = np.abs(audio_data).max()
            
            # Dynamic noise gate:
            # - Lower threshold when crying detected (let audio through)
            # - Higher threshold for background noise
            if is_crying:
                noise_threshold = 200  # More permissive during crying
            else:
                noise_threshold = 500  # Standard threshold
            
            # If sound is below threshold, mute it (send zeros)
            if peak < noise_threshold:
                data = b'\x00' * len(data)
            
            yield data
        except Exception as e:
            print(f"Stream error: {e}")
            break
    
    stream.stop_stream()
    stream.close()
    p.terminate()

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/audio_feed')
def audio_feed():
    return Response(gen_audio(), mimetype="application/octet-stream")


# Legacy motion status endpoint (for backward compatibility)
@app.route('/motion_status')
def motion_status():
    return jsonify({'motion': motion_detected})


# =============================================================================
# ML API ENDPOINTS
# =============================================================================

@app.route('/api/status')
def api_status():
    """Get combined status of all ML systems."""
    with audio_ml_lock:
        audio = audio_ml_result.copy()
    with motion_ml_lock:
        motion = motion_ml_result.copy()
    
    alert = alert_manager.get_status()
    
    return jsonify({
        "audio": {
            "is_crying": audio.get("is_crying", False),
            "confidence": round(audio.get("confidence", 0), 3),
            "detected_class": audio.get("detected_class", ""),
            "alert_level": audio.get("alert_level", "none")
        },
        "motion": {
            "detected": motion.get("motion_detected", False),
            "type": motion.get("motion_type", "none"),
            "confidence": round(motion.get("confidence", 0), 3),
            "breathing_detected": motion.get("breathing_detected", False),
            "breathing_rate": motion.get("breathing_rate", 0)
        },
        "alert": alert,
        "timestamp": time.time()
    })


@app.route('/api/audio_status')
def api_audio_status():
    """Get audio classifier status."""
    return jsonify(audio_classifier.get_status())


@app.route('/api/motion_status')
def api_motion_status():
    """Get motion detector status."""
    return jsonify(motion_detector.get_status())


@app.route('/api/alert_status')
def api_alert_status():
    """Get alert manager status."""
    return jsonify(alert_manager.get_status())


@app.route('/api/alert_history')
def api_alert_history():
    """Get recent alert history."""
    limit = request.args.get('limit', 10, type=int)
    return jsonify(alert_manager.get_history(limit))


@app.route('/api/acknowledge', methods=['POST'])
def api_acknowledge():
    """Acknowledge current alert."""
    alert_manager.acknowledge_alert()
    return jsonify({"status": "acknowledged"})


@app.route('/api/sensitivity', methods=['GET', 'POST'])
def api_sensitivity():
    """Get or set sensitivity level."""
    if request.method == 'POST':
        data = request.get_json()
        level = data.get('level', 'medium')
        if level in ['low', 'medium', 'high']:
            alert_manager.set_sensitivity(level)
            return jsonify({"status": "updated", "level": level})
        return jsonify({"error": "Invalid sensitivity level"}), 400
    
    return jsonify({"level": alert_manager.sensitivity})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset all ML systems."""
    audio_classifier.reset()
    motion_detector.reset()
    alert_manager.reset()
    return jsonify({"status": "reset complete"})


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("BABY MONITOR WITH ML - Starting up...")
    print("="*60)
    print("\nML Features:")
    print("  - Audio: YAMNet-based infant cry detection")
    print("  - Motion: Adaptive background subtraction with pattern analysis")
    print("  - Alerts: Multi-signal fusion with false positive suppression")
    print("\nAPI Endpoints:")
    print("  - /api/status        - Combined ML status")
    print("  - /api/alert_history - Recent alerts")
    print("  - /api/sensitivity   - Get/set sensitivity (low/medium/high)")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
