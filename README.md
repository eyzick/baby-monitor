# My DIY Baby Monitor

I built this simple baby monitor to stream video and audio from my Raspberry Pi to my phone (or any device on the network). It works great for keeping an ear and eye on things!

## What You'll Need

To get this working, I used:
- A Raspberry Pi (any model with Wi-Fi/Ethernet should work)
- A Camera Module or USB Webcam
- A USB Microphone (or an audio HAT)
- Python 3

## Getting It Running

Here is how I set it up on my Pi:

1.  **Get the code:** First, clone this repo or copy these files onto your Raspberry Pi.

2.  **Audio stuff (Important!):**
    I found out the hard way that you need a specific system library for the audio to work. Run this in your terminal first:
    
    ```bash
    sudo apt-get update
    sudo apt-get install portaudio19-dev
    ```

3.  **Install Python packages:**
    Now, install the python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    *Tip: If you're on a newer Raspberry Pi OS, it might complain about environments. If so, just make a virtual environment like this:*
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

## How to Use It

1.  Start the app:

    ```bash
    python3 app.py
    ```

2.  It usually runs on port 5000.

3.  Find your Pi's IP address (I usually run `hostname -I` to check mine).

4.  Grab your phone or laptop, open the browser, and type in the address:

    ```
    http://<YOUR_PI_IP_ADDRESS>:5000
    ```
    (Just replace `<YOUR_PI_IP_ADDRESS>` with your actual IP, like `192.168.1.15`)

## If Things Go Wrong

Here are a few issues I ran into and how I fixed them:

### No Audio on My Phone?
Most mobile browsers (like Chrome or Safari) block audio if the site isn't secure (HTTPS). Since this runs locally on `http`, they might block it.
- **Fix:** I used Firefox on my phone, or I went into Chrome flags (`chrome://flags/#unsafely-treat-insecure-origin-as-secure`) and added my Pi's IP to the allowed list.

### Camera Issues?
If the screen is black:
- Check your connections.
- On newer Pi OS versions, I had to enable "Legacy Camera" support in `sudo raspi-config`.
- If you're using a USB webcam instead of the Pi camera, you might need to change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the code (`app.py`).

### Is it Laggy?
If the video is stuttering, I usually lower the resolution. You can add this to `app.py` right after the camera starts:
```python
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```
