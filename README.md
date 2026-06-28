# Real-Time Object Recognition (YOLO26n)

A real-time object recognition system built using **YOLO26n (Ultralytics)**, **OpenCV**, **PyTorch**, and **SQLite**. The application performs live webcam inference, displays object detections with confidence scores, logs every detection to a local database, and optionally announces newly detected objects using text-to-speech.

This project serves as the perception module for a larger humanoid robotics pipeline and is designed for future deployment on edge AI hardware such as the **Raspberry Pi 5 + Hailo AI HAT+**.

---

## Features

- Real-time object detection using YOLO26n
- Live webcam inference with bounding boxes and confidence scores
- GPU/CPU automatic device selection
- Real-time FPS display
- SQLite database logging of every detected object
- Timestamp, object class, and confidence stored for each detection
- Optional voice announcements using `pyttsx3`
- Detection cooldown to prevent repeated announcements
- Save image snapshots during runtime
- Session summary showing object detection statistics

---

## Demo

During execution the application displays:

- Live webcam feed
- Bounding boxes around detected objects
- Class labels
- Confidence scores
- FPS counter
- Current inference device (CPU/GPU)

Example output:

```
Person 0.94
Laptop 0.89
Bottle 0.81

FPS : 42.3
Device : CUDA
```

---

## Project Structure

```
Object-Recognition/
│
├── snapshots/                 # Saved screenshots
├── yolo-env/                  # Virtual environment (optional)
│
├── object_recognition_demo.py # Main application
├── detections.db              # SQLite detection log
├── yolo26n.pt                 # YOLO26n model
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python 3 |
| Object Detection | YOLO26n (Ultralytics) |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV |
| Database | SQLite |
| Voice Feedback | pyttsx3 |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Object-Recognition-yolo26.git
cd Object-Recognition-yolo26
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv yolo-env
yolo-env\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv yolo-env
source yolo-env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If using NVIDIA CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Usage

Run the application:

```bash
python object_recognition_demo.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| **S** | Save snapshot |
| **Q** | Quit application |

---

## Database Logging

Every detected object is stored in **detections.db**.

Each record contains:

- Detection ID
- Timestamp
- Object class
- Confidence score

Example:

| Timestamp | Object | Confidence |
|-----------|---------|------------|
| 2026-06-28 09:51:04 | person | 0.91 |
| 2026-06-28 09:51:05 | laptop | 0.88 |
| 2026-06-28 09:51:07 | bottle | 0.84 |

---

## Voice Feedback

When enabled, the system announces newly detected objects using **pyttsx3**.

Example:

```
"Person detected"
"Laptop detected"
```

Repeated announcements are automatically throttled to avoid continuous repetition.

---

## Snapshots

Press:

```
S
```

to save the current frame.

Images are stored inside:

```
snapshots/
```

---

## Session Summary

When the application exits, a summary similar to the following is displayed:

```
========== SESSION SUMMARY ==========

person : 87

chair : 23

laptop : 18

bottle : 12

=====================================
```

---

## Future Improvements

- Multi-object tracking
- Object counting
- Region-of-interest detection
- Face recognition integration
- ONNX export
- Hailo AI HAT+ deployment
- Raspberry Pi 5 optimization
- Humanoid robot perception integration

---

## Requirements

- Python 3.10+
- Webcam
- OpenCV
- PyTorch
- Ultralytics
- SQLite3

---

## License

This project is licensed under the MIT License.

---

## Author

**Rifa V P Zacharia Kassim**

GitHub: https://github.com/rvpzacharia-ai
