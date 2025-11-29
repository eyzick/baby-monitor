from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import numpy as np
import time
import pyaudio
import struct

app = Flask(__name__)

# Video Settings
camera = cv2.VideoCapture(0)

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 4096 

# Noise Gate Settings
NOISE_THRESHOLD = 500  # Amplitude threshold (0-32768). Adjust if too sensitive/insensitive.

# Motion Detection Settings
motion_detected = False
last_motion_time = 0

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

def generate_frames():
    global motion_detected, last_motion_time
    
    # Read first frame for reference
    success, prev_frame = camera.read()
    if success:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    else:
        prev_gray = None

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

        # Motion detection logic
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            # Compute difference between current frame and previous frame
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Dilate the thresholded image to fill in holes
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            current_motion = False
            for contour in contours:
                if cv2.contourArea(contour) < 5000: 
                    continue
                current_motion = True
            
            if current_motion:
                motion_detected = True
                last_motion_time = time.time()
            elif time.time() - last_motion_time > 2.0:
                motion_detected = False

        # Update previous frame
        prev_gray = gray

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_data = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

def gen_audio():
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

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Noise Gate Logic
            # Convert bytes to numpy array of int16
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate peak amplitude
            peak = np.abs(audio_data).max()
            
            # If sound is below threshold, mute it (send zeros)
            if peak < NOISE_THRESHOLD:
                data = b'\x00' * len(data)
            
            yield data
        except Exception as e:
            print(f"Stream error: {e}")
            break
    
    stream.stop_stream()
    stream.close()
    p.terminate()

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

@app.route('/motion_status')
def motion_status():
    return jsonify({'motion': motion_detected})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
