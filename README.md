# Real-Time Object Recognition (YOLO26n)

Real-time object detection demo using **YOLO26n** (Ultralytics, 2026's edge-first NMS-free architecture), OpenCV, SQLite logging, and optional voice feedback. Built as Phase 1 of a larger pipeline — this same model/workflow is designed to later be exported and deployed on a **Raspberry Pi 5 + Hailo NPU** for integration into a humanoid robot's perception stack.

## Features
- Live webcam object detection (80 COCO classes) with bounding boxes + confidence scores
- Real-time FPS overlay, GPU/CPU device indicator
- Every detection logged to a SQLite database (`detections.db`) with timestamp, class, confidence
- Optional spoken announcements (`pyttsx3`) for newly detected objects, debounced to avoid spam
- Snapshot capture on keypress
- End-of-session summary: per-class detection counts

## Tech Stack
| Component | Tool |
|---|---|
| Detection model | YOLO26n (Ultralytics) |
| Inference backend | PyTorch (CUDA on NVIDIA GPU, CPU fallback) |
| Video I/O | OpenCV |
| Logging | SQLite |
| Voice feedback | pyttsx3 |

## Setup

```bash
python -m venv yolo-env
yolo-env\Scripts\activate          # Windows
# source yolo-env/bin/activate      # Linux/Mac

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Usage

```bash
python object_recognition_demo.py
```

- Press **q** to quit
- Press **s** to save a snapshot of the current frame

On exit, a session summary is printed and the full detection log is available in `detections.db`.

## Roadmap
- [ ] Export model to ONNX → HEF for Hailo NPU deployment
- [ ] Deploy on Raspberry Pi 5 + AI HAT+ for real-time inference on a humanoid robot
- [ ] Integrate with existing face recognition module (MediaPipe BlazeFace + MobileFaceNet)
- [ ] Add servo-driven object tracking (pan-tilt head)

## License
MIT
