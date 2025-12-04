# 👁️ Eye Strain Monitor - Blink Detection

A real-time eye strain monitoring application that tracks your blink rate and distance from the screen using computer vision to help prevent eye fatigue during long screen sessions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

### 🔍 Real-Time Monitoring
- **Blink Detection**: Tracks eye blinks using Eye Aspect Ratio (EAR) algorithm
- **Distance Measurement**: Monitors your distance from screen using iris tracking
- **Live Metrics**: Displays blink rate per minute and total blinks

### ⚠️ Smart Notifications
- Alerts when blink rate falls below healthy levels (15 blinks/min recommended)
- Warns if you sit too close to the screen (\<30cm) for extended periods
- Desktop notifications to remind you to take breaks

### 🎯 Calibration System
- One-click calibration for accurate distance measurement
- Customizable reference distance (default: 50cm)

### 💻 Two Modes Available

#### Desktop App (`blink_app.py`)
- **PyQt5 GUI** - Clean, minimal interface
- **Background monitoring** - Works even when minimized
- **Offline** - No internet required
- **Privacy-first** - All processing happens locally

#### Web App (`index.html`)
- **Browser-based** - No installation required
- **Modern UI** - Sleek dark theme with glassmorphism
- **MediaPipe WASM** - Runs entirely in browser
- **[Try it live](https://your-github-username.github.io/strain_analysis)** 🚀

## 📋 Prerequisites

- **Python 3.9 or higher**
- **Webcam**
- **Windows/Mac/Linux**

## 🚀 Quick Start

### Using UV (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Clone the repository
git clone https://github.com/your-username/strain_analysis.git
cd strain_analysis

# Install uv if you haven't already
pip install uv

# Create virtual environment and install dependencies
uv sync

# Run the desktop app
uv run python blink_app.py
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/strain_analysis.git
cd strain_analysis

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the desktop app
python blink_app.py
```

### Web Version

Simply open `index.html` in your browser or visit the [GitHub Pages deployment](https://your-username.github.io/strain_analysis).

## 📦 Dependencies

```
opencv-python>=4.5.0  # Computer vision
mediapipe>=0.10.0     # Face mesh detection
PyQt5>=5.15.0         # GUI framework
plyer>=2.0.0          # Cross-platform notifications
numpy>=1.19.0         # Numerical operations
```

## 🎮 How to Use

### Desktop App

1. **Launch the application**
   ```bash
   python blink_app.py
   ```

2. **Allow camera access** when prompted

3. **Set your distance**
   - Enter your current distance from screen (in cm)
   - Default: 50cm (recommended)

4. **Calibrate**
   - Sit at your measured distance
   - Look straight at the camera
   - Click "Calibrate" button

5. **Monitor your eyes!**
   - App will track blinks automatically
   - Notifications appear for eye strain warnings
   - Works even when minimized or using other apps

### Health Recommendations

| Metric | Healthy Range | Warning Level |
|--------|--------------|---------------|
| **Blinks/min** | 15+ | \<15 (Low), \<5 (High Risk) |
| **Screen Distance** | 50-100cm | \<30cm or \>100cm |
| **Break Frequency** | Every 20 min | - |

**Follow the 20-20-20 rule:**  
Every 20 minutes, look at something 20 feet (6 meters) away for 20 seconds.

## 🏗️ Project Structure

```
strain_analysis/
├── blink_app.py              # Desktop PyQt5 application
├── index.html                # Web version (MediaPipe WASM)
├── app.html                  # Alternative web version
├── requirements.txt          # Python dependencies
├── pyproject.toml           # UV project configuration
├── icon.ico                 # Application icon
└── README.md                # This file
```

## 🔧 Technical Details

### Blink Detection Algorithm

Uses **Eye Aspect Ratio (EAR)**:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Where p1-p6 are eye landmark points from MediaPipe Face Mesh.

- **Threshold**: EAR < 0.27
- **Consecutive Frames**: 3 frames (~80ms)
- **Prevents double-counting**: 150ms cooldown

### Distance Measurement

Uses proportional relationship:
```
Distance = K / iris_width_pixels
```

Where K is calibration constant = known_distance × iris_pixels_at_calibration

### MediaPipe Face Mesh
- 478 facial landmarks
- 4 iris landmarks per eye (468-477)
- Real-time tracking at 30+ FPS

## 🌐 Web vs Desktop

| Feature | Web (`index.html`) | Desktop (`blink_app.py`) |
|---------|-------------------|-------------------------|
| **Installation** | None - just open | Python + dependencies |
| **Background Monitoring** | ❌ Stops when tab inactive | ✅ Works when minimized |
| **Internet Required** | ✅ For MediaPipe CDN | ❌ Fully offline |
| **Privacy** | ✅ On-device only | ✅ Local processing |
| **Notifications** | ⚠️ Browser notifications | ✅ System notifications |
| **Best For** | Quick demos, testing | Daily use, serious monitoring |

## 🐛 Troubleshooting

### Camera Not Working

**Windows:**
```
Settings > Privacy > Camera > Allow desktop apps
```

**Mac:**
```
System Preferences > Security & Privacy > Camera
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### MediaPipe Issues

```bash
# Use specific compatible versions
pip install mediapipe==0.10.9 opencv-python==4.8.0.76
```

## 📝 License

MIT License - feel free to use this project for personal or commercial purposes.

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 💡 Future Enhancements

- [ ] Export blink rate data to CSV
- [ ] Weekly/monthly eye health reports
- [ ] Customizable notification sounds
- [ ] Multi-monitor support
- [ ] Dark mode for desktop app
- [ ] Mobile app version

## 🙏 Acknowledgments

- **MediaPipe** by Google for face mesh detection
- **OpenCV** for computer vision capabilities
- **PyQt5** for the desktop GUI framework

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⚠️ Disclaimer:** This app is for monitoring purposes only and is not a medical device. Consult an eye care professional for persistent eye strain or vision problems.

**Made with ❤️ for healthier screen time**
