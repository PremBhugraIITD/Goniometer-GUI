# Goniometer-GUI

A comprehensive web-based graphical user interface for automated goniometer measurements, enabling precise contact angle analysis and surface tension calculations using advanced computer vision techniques.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![React](https://img.shields.io/badge/React-18.3.1-blue)
![Node.js](https://img.shields.io/badge/Node.js-Express-green)
![Python](https://img.shields.io/badge/Python-OpenCV-yellow)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Analysis Modes](#analysis-modes)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Hardware Integration](#hardware-integration)
- [Scientific Background](#scientific-background)
- [Contributing](#contributing)
- [License](#license)

## 🔬 Overview

Goniometer-GUI is a sophisticated full-stack application designed for scientific measurement of:
- **Contact Angles** (Static, Advancing, Receding)
- **Surface Tension** using the Pendant Drop method
- **Contact Angle Hysteresis**
- **Device Calibration**

The system combines real-time camera feed processing, automated image analysis, and hardware control into a unified, user-friendly interface suitable for research in surface chemistry, materials science, and fluid dynamics.

## ✨ Features

### Core Capabilities
- 🎥 **Real-time Camera Feed** - Live video capture from browser or connected devices
- 📸 **Interactive Image Capture** - Screenshot and video recording capabilities
- 🔍 **Advanced Image Processing** - OpenCV-based edge detection and contour analysis
- 📊 **Live Data Visualization** - Real-time plotting of measurements
- 💾 **Data Export** - CSV export for detailed analysis
- 🎬 **Annotated Videos** - Processed videos with visual overlays
- 🎛️ **Hardware Control** - IoT-based motor control for positioning
- 🔧 **Calibration System** - Pixel-to-metric conversion accuracy

### Analysis Features
- Interactive ROI (Region of Interest) selection
- Adjustable Canny edge detection thresholds
- Baseline and vertical line selection using sliders
- Polynomial curve fitting for contact angle calculation
- Multi-frame video analysis with live graphs
- Automatic advancing/receding angle detection

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client (React + Vite)                    │
│  ┌──────────────┬──────────────┬──────────────┬───────────┐ │
│  │ VideoSection │ Controller   │ Results      │ Hardware  │ │
│  │ (Camera)     │ Section      │ Section      │ Control   │ │
│  └──────────────┴──────────────┴──────────────┴───────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↕ HTTP/SSE
┌─────────────────────────────────────────────────────────────┐
│              Backend (Node.js/Express Server)                │
│  • File Upload Handling (Multer)                            │
│  • Python Script Orchestration                               │
│  • Server-Sent Events for Real-time Updates                 │
└─────────────────────────────────────────────────────────────┘
                            ↕ Spawn
┌─────────────────────────────────────────────────────────────┐
│                 Python Analysis Engine                       │
│  • Sessile Drop Analysis                                     │
│  • Pendant Drop Analysis (Image & Video)                     │
│  • Hysteresis Analysis                                       │
│  • Calibration Processing                                    │
└─────────────────────────────────────────────────────────────┘
```

## 🔬 Analysis Modes

### 1. **Sessile Drop Contact Angle**
Measures the static contact angle of a liquid droplet resting on a solid surface.

**Algorithm:**
- Interactive image cropping
- Adjustable edge detection thresholds
- Baseline selection via slider
- Contour detection and polynomial fitting
- Tangent slope calculation at baseline intersection
- Supports both acute (<90°) and obtuse (>90°) angles

**Output:** Left, Right, and Average contact angles

### 2. **Pendant Drop Surface Tension (Image)**
Calculates surface tension from a single image of a pendant drop.

**Algorithm:**
- Measures maximum diameter (d_e) at widest point
- Measures secondary diameter (d_s) at height = (lowest point - d_e)
- Detects and measures needle diameter for calibration
- Applies Young-Laplace equation variant

**Formula:**
```
γ = (Δρ × g × d_e^4.49 × 0.35) / d_s^2.49
```

**Output:** d_e, d_s, Surface Tension (mN/m)

### 3. **Pendant Drop Surface Tension (Video)**
Real-time surface tension measurement from video frames.

**Features:**
- Frame-by-frame analysis
- Live matplotlib visualization
- Annotated output video generation
- CSV export with all frame data

**Output:** Time-series surface tension data, processed video

### 4. **Contact Angle Hysteresis**
Measures the difference between advancing and receding contact angles.

**Algorithm:**
- Vertical line selection to mask needle
- Width tracking across video frames
- Peak detection for advancing phase (max width increase)
- Trough detection for receding phase (max width decrease)
- Contact angle calculation for each frame

**Formula:**
```
Hysteresis = θ_advancing - θ_receding
```

**Output:** Advancing CA, Receding CA, Hysteresis value, CSV data

### 5. **Calibration**
Establishes pixel-to-millimeter conversion using a known needle diameter.

**Process:**
- Uses known needle diameter (1.0194 mm)
- Measures droplet parameters
- Calculates corrected calibration factor

**Formula:**
```
C_n_d = √((21.292282 × d_s^2.49 × d_n^2) / d_e^4.49)
```

**Output:** Calibrated conversion factor saved to file

## 📦 Prerequisites

### Software Requirements
- **Node.js** (v14 or higher)
- **Python** 3.8+
- **npm** or **yarn**
- Modern web browser with camera access

### Python Dependencies
```bash
pip install opencv-python numpy matplotlib scipy
```

### Node.js Dependencies
See `package.json` files in `/client` and `/python` directories.

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/PremBhugraIITD/Goniometer-GUI.git
cd Goniometer-GUI
```

### 2. Install Client Dependencies
```bash
cd client
npm install
```

### 3. Install Server Dependencies
```bash
cd ../python
npm install
```

### 4. Install Python Dependencies
```bash
pip install opencv-python numpy matplotlib scipy
```

## 💻 Usage

### Starting the Application

**Terminal 1 - Start the Backend Server:**
```bash
cd python
node index.js
```
Server runs on `http://localhost:3000`

**Terminal 2 - Start the React Frontend:**
```bash
cd client
npm run dev
```
Frontend runs on `http://localhost:5173`

### Workflow

1. **Open the Application** in your browser at `http://localhost:5173`

2. **Select Analysis Mode** from the dropdown menu:
   - Sessile Drop
   - Pendant Drop (Image)
   - Pendant Drop (Video)
   - Hysteresis
   - Calibration

3. **Configure Parameters** (if required):
   - Enter liquid density for Pendant Drop analyses

4. **Capture/Record**:
   - Click "Capture Screenshot" for image-based analysis
   - Click "Start Recording" / "Stop Recording" for video-based analysis

5. **Interactive Processing**:
   - Crop the region of interest
   - Adjust Canny edge detection thresholds
   - Select baseline (for contact angle)
   - Select vertical lines (for hysteresis)

6. **View Results** in real-time in the Results section

7. **Download Data** as CSV for further analysis

8. **Control Hardware** (optional) using the XYZ control panel

## 📡 API Documentation

### Endpoints

#### Analysis Endpoints

**POST `/sessile-drop`**
- Upload image for sessile drop analysis
- Body: `multipart/form-data` with `image` field
- Returns: Success message

**POST `/pendant-drop-image`**
- Upload image and density for pendant drop analysis
- Body: `multipart/form-data` with `image` and `density` fields
- Returns: Success message

**POST `/pendant-drop-video`**
- Upload video and density for pendant drop video analysis
- Body: `multipart/form-data` with `video` and `density` fields
- Returns: Success message

**POST `/hysteresis-analysis`**
- Upload video for hysteresis analysis
- Body: `multipart/form-data` with `video` field
- Returns: Success message

**POST `/calibration`**
- Upload image for calibration
- Body: `multipart/form-data` with `image` field
- Returns: Success message

#### Real-time Streaming (Server-Sent Events)

**GET `/sessile-drop-stream`**
- Streams real-time updates from `Static_Contact_Angle.txt`

**GET `/pendant-drop-stream`**
- Streams real-time updates from `Surface_Tension.txt`

**GET `/hysteresis-stream`**
- Streams real-time updates from `Contact_Angle_Hysteresis.txt`

**GET `/calibration-stream`**
- Streams real-time updates from `Calibration_result.txt`

#### Data Export

**GET `/download-results-hysteresis`**
- Downloads `contact_angle_data.csv`

**GET `/download-results-pendant`**
- Downloads `Surface_Tension_data.csv`

#### Utility

**POST `/run-scrcpy`**
- Launches scrcpy for Android device screen mirroring
- Requires scrcpy installed on system

## 🔧 Hardware Integration

### IoT Control via Blynk
The system integrates with Blynk IoT platform for motor control:

- **X-axis**: ±0.5 mm increments
- **Y-axis**: ±1 mm increments
- **Z-axis**: ±1 mm increments
- **Extruder**: ±5 mm increments

G-code commands are sent via Blynk API for precise positioning control.

### Android Device Integration
- Uses **scrcpy** for wireless screen mirroring
- Enables high-quality camera feed from smartphone
- Requires USB debugging enabled on Android device

## 📚 Scientific Background

### Contact Angle Measurement
Contact angle (θ) is the angle formed by a liquid at the three-phase boundary where liquid, gas, and solid intersect. It quantifies the wettability of a solid surface.

- **θ < 90°**: Hydrophilic (wetting)
- **θ > 90°**: Hydrophobic (non-wetting)

### Contact Angle Hysteresis
The difference between advancing and receding contact angles, indicating surface heterogeneity and roughness.

### Surface Tension (Pendant Drop Method)
Based on the Young-Laplace equation, relating the pressure difference across a curved interface to surface tension and curvature. The pendant drop method analyzes the shape of a hanging droplet to calculate surface tension.

### Mathematical Models

**Contact Angle Calculation:**
1. Polynomial fitting: `y = ax² + bx + c`
2. Derivative: `dy/dx = 2ax + b`
3. Angle: `θ = arctan(slope) × 180/π`

**Surface Tension:**
```
γ = (Δρ × g × d_e^4.49 × 0.35) / d_s^2.49
```
Where:
- γ = surface tension
- Δρ = density difference between liquid and air
- g = gravitational acceleration (9.81 m/s²)
- d_e = maximum diameter
- d_s = secondary diameter

## 🛠️ Technology Stack

### Frontend
- **React** 18.3.1 - UI framework
- **Vite** - Build tool
- **Axios** - HTTP client
- **React Router DOM** - Navigation
- **CSS Grid** - Layout system

### Backend
- **Node.js** - Runtime
- **Express** 4.21.2 - Web framework
- **Multer** - File upload handling
- **CORS** - Cross-origin resource sharing
- **Morgan** - HTTP request logger

### Image Processing
- **OpenCV (cv2)** - Computer vision
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization & interactive widgets
- **SciPy** - Scientific computing (interpolation, signal processing)

## 📁 Project Structure

```
Goniometer-GUI/
├── client/                      # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoSection/   # Camera feed & capture
│   │   │   ├── ControllerSection/  # Mode selection
│   │   │   ├── ResultsSection/     # Real-time results
│   │   │   ├── InputSection/       # Hardware control
│   │   │   ├── Navbar/
│   │   │   └── Footer/
│   │   ├── pages/
│   │   │   └── Home/           # Main page
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── public/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── python/                      # Backend & analysis
│   ├── index.js                # Express server
│   ├── Sessile_ossila_algo.py  # Sessile drop analysis
│   ├── pendant_ST.py           # Pendant drop (image)
│   ├── video_pendant_drop.py   # Pendant drop (video)
│   ├── hysteresis_final.py     # Hysteresis analysis
│   ├── Calibration.py          # Calibration
│   ├── Thresholds.py           # Threshold adjustment utility
│   ├── package.json
│   └── [output files]          # Generated results
│
└── README.md
```

## 🎯 Use Cases

- **Materials Research**: Surface characterization of coatings, polymers, and composites
- **Quality Control**: Wettability testing in manufacturing
- **Pharmaceutical**: Drug formulation and delivery systems
- **Biomedical**: Biomaterial compatibility testing
- **Chemical Engineering**: Surfactant and detergent development
- **Nanotechnology**: Superhydrophobic/superhydrophilic surface analysis

## 🔍 Troubleshooting

### Camera Not Detected
- Ensure browser has camera permissions
- Check if another application is using the camera
- Try using scrcpy for external camera feed

### Python Script Errors
- Verify Python path in `index.js` matches your installation
- Check all Python dependencies are installed
- Ensure file paths are correctly configured

### Real-time Updates Not Showing
- Check if backend server is running on port 3000
- Verify Server-Sent Events are not blocked by firewall
- Check browser console for connection errors

### CSV Download Not Working
- Ensure Python script completed successfully
- Check if output files are generated in python directory
- Verify file permissions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Prem Bhugra** - [PremBhugraIITD](https://github.com/PremBhugraIITD)

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- React and Vite teams for excellent development experience
- Scientific community for established measurement methodologies
- IIT Delhi for academic support

## 📧 Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the maintainers.

---

**Note**: This application is designed for research and educational purposes. Ensure proper calibration and validation for critical measurements.
