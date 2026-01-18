# 👶 My DIY Baby Monitor - ML Enhanced

I built this baby monitor to stream video and audio from my Raspberry Pi to my phone. Now with **Machine Learning** for intelligent cry detection and motion analysis!

## ✨ ML Features

### 🔊 Audio Intelligence
- **YAMNet-based Cry Detection**: Uses Google's YAMNet model to distinguish infant cries from background noise
- **Multi-class Recognition**: Detects crying, whimpering, screaming, and other baby sounds
- **Adaptive Thresholding**: Automatically adjusts sensitivity based on ambient noise levels
- **Temporal Smoothing**: Reduces false positives by analyzing patterns over time

### 📹 Smart Motion Detection
- **ML-Enhanced Analysis**: Goes beyond simple frame differencing with pattern classification
- **Motion Type Classification**: Distinguishes between noise, lighting changes, small movements, and significant activity
- **Breathing Detection**: Monitors subtle periodic motion to detect breathing patterns
- **No-Motion Alerts**: Warns if baby hasn't moved for extended periods

### 🔔 Intelligent Alerts
- **Multi-Signal Fusion**: Combines audio and motion data for higher accuracy
- **Configurable Sensitivity**: Choose between Low, Medium, or High sensitivity profiles
- **False Positive Suppression**: Requires confirmation across multiple frames before alerting
- **Alert History**: Track past alerts with timestamps and confidence levels

## 🛠️ What You'll Need

- A Raspberry Pi (any model with Wi-Fi/Ethernet)
- A Camera Module or USB Webcam
- A USB Microphone (or an audio HAT)
- Python 3.9+

## 🚀 Getting It Running

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd baby_monitor
```

### 2. Install System Dependencies

The audio library needs PortAudio:

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev
```

### 3. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Python Packages

```bash
pip install -r requirements.txt
```

**Note for Raspberry Pi**: For better performance, use TensorFlow Lite instead of full TensorFlow:

```bash
# Comment out tensorflow in requirements.txt and instead install:
pip install tflite-runtime
```

### 5. First Run - Model Download

On first run, the YAMNet model will be automatically downloaded (~3MB). This requires internet access.

## 📱 How to Use It

1. Start the application:

```bash
python3 app.py
```

2. You'll see startup information:

```
============================================================
BABY MONITOR WITH ML - Starting up...
============================================================

ML Features:
  - Audio: YAMNet-based infant cry detection
  - Motion: Adaptive background subtraction with pattern analysis
  - Alerts: Multi-signal fusion with false positive suppression

API Endpoints:
  - /api/status        - Combined ML status
  - /api/alert_history - Recent alerts
  - /api/sensitivity   - Get/set sensitivity (low/medium/high)
============================================================
```

3. Find your Pi's IP address:

```bash
hostname -I
```

4. Open a browser and navigate to:

```
http://<YOUR_PI_IP_ADDRESS>:5000
```

## 🎛️ Configuration

### Sensitivity Profiles

| Profile | Audio Threshold | Motion Threshold | Best For |
|---------|----------------|------------------|----------|
| **Low** | 0.6 | 0.7 | Noisy environments, reduces false alerts |
| **Medium** | 0.4 | 0.5 | Balanced - recommended default |
| **High** | 0.3 | 0.4 | Quiet rooms, maximum sensitivity |

You can change sensitivity through the web UI or via API:

```bash
curl -X POST http://localhost:5000/api/sensitivity \
  -H "Content-Type: application/json" \
  -d '{"level": "high"}'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Combined ML status (audio, motion, alerts) |
| `/api/audio_status` | GET | Detailed audio classifier status |
| `/api/motion_status` | GET | Detailed motion detector status |
| `/api/alert_status` | GET | Alert manager status and stats |
| `/api/alert_history` | GET | Recent alert history |
| `/api/acknowledge` | POST | Acknowledge current alert |
| `/api/sensitivity` | GET/POST | Get or set sensitivity level |
| `/api/reset` | POST | Reset all ML systems |

## 🔧 Troubleshooting

### No Audio on Mobile?
Most mobile browsers block audio from non-HTTPS sites.
- **Fix**: Use Firefox, or add the Pi's IP to Chrome's secure origins in `chrome://flags/#unsafely-treat-insecure-origin-as-secure`

### Camera Issues?
- Check connections
- Enable "Legacy Camera" in `sudo raspi-config` for Pi Camera
- For USB webcam, try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app.py`

### ML Model Not Loading?
- Ensure internet access on first run for model download
- Check that TensorFlow or tflite-runtime is installed correctly
- On Raspberry Pi, tflite-runtime is recommended over full TensorFlow

### High CPU Usage?
- Lower video resolution by adding to `app.py`:
  ```python
  camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  ```
- Use "Low" sensitivity to reduce ML processing frequency
- On Raspberry Pi, ensure you're using tflite-runtime

### Too Many False Alerts?
- Switch to "Low" sensitivity profile
- The system learns ambient noise over time - give it a few minutes
- Ensure camera is stable (tripod/mount) to reduce motion noise

### Missing Alerts?
- Switch to "High" sensitivity profile
- Check that the microphone is working (`arecord -l`)
- Verify the baby is within the camera's field of view

## 📁 Project Structure

```
baby_monitor/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── ml/                   # ML modules
│   ├── __init__.py
│   ├── audio_classifier.py   # YAMNet-based cry detection
│   ├── motion_detector.py    # Smart motion analysis
│   └── alert_manager.py      # Alert fusion and management
├── static/
│   └── style.css         # UI styling
└── templates/
    └── index.html        # Web interface
```

## 🧠 How the ML Works

### Audio Classification
1. Audio is captured at 44.1kHz and resampled to 16kHz for ML
2. ~1 second chunks are fed to YAMNet (521-class audio classifier)
3. Baby cry classes are extracted and scored
4. Temporal smoothing averages predictions over 5 frames
5. Adaptive thresholding adjusts based on ambient noise

### Motion Detection
1. Frames are converted to grayscale and blurred
2. MOG2 background subtraction isolates moving objects
3. Contour analysis classifies motion patterns
4. Shadow detection prevents lighting false positives
5. Breathing detection uses FFT to find periodic motion (9-30 bpm)

### Alert Fusion
1. Audio and motion signals are combined
2. Confirmation requires consistent detection over time
3. Cooldown prevents alert spam
4. Priority levels (low/medium/high/critical) based on confidence

## 📜 License

MIT License - feel free to use and modify!
