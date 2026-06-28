"""
================================================================
 Real-Time Object Recognition Demo
 Model   : YOLO26n (Ultralytics, Jan 2026 release - edge-first, NMS-free)
 Backend : PyTorch (CUDA on RTX 3050 if available, else CPU)
 Extras  : Live bounding boxes, FPS counter, SQLite detection log,
           optional voice announcements (pyttsx3), snapshot saving

 RUN:  python object_recognition_demo.py
 QUIT: press 'q' in the video window
 SNAP: press 's' to save the current annotated frame
================================================================
"""

import os
import time
import sqlite3
import threading
from datetime import datetime
from collections import defaultdict

import cv2
import torch
from ultralytics import YOLO

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# ---------------------------- CONFIG ----------------------------
MODEL_PATH = "yolo26n.pt"      # auto-downloads on first run (~tiny, edge-optimized)
CONFIDENCE_THRESHOLD = 0.5
CAMERA_INDEX = 0               # 0 = default webcam; try 1 if you have an external cam
DB_PATH = "detections.db"
ENABLE_VOICE = True            # set False to disable spoken announcements
VOICE_COOLDOWN_SEC = 5         # don't re-announce the same class within this window
SNAPSHOT_DIR = "snapshots"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# ------------------------------------------------------------------

os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def init_db(path):
    """Creates the SQLite log table if it doesn't already exist."""
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            class_name  TEXT NOT NULL,
            confidence  REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


class VoiceAnnouncer:
    """Speaks newly-detected object classes without spamming, in a background thread."""

    def __init__(self, enabled=True, cooldown=5):
        self.enabled = enabled and PYTTSX3_AVAILABLE
        self.cooldown = cooldown
        self.last_said = {}
        self.engine = None
        if self.enabled:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 170)
            except Exception as e:
                print(f"[VOICE] Disabled (init failed): {e}")
                self.enabled = False
        elif enabled and not PYTTSX3_AVAILABLE:
            print("[VOICE] pyttsx3 not installed, voice feedback disabled.")

    def announce(self, class_name):
        if not self.enabled:
            return
        now = time.time()
        if class_name in self.last_said and (now - self.last_said[class_name]) < self.cooldown:
            return
        self.last_said[class_name] = now
        threading.Thread(target=self._speak, args=(class_name,), daemon=True).start()

    def _speak(self, class_name):
        try:
            self.engine.say(f"{class_name} detected")
            self.engine.runAndWait()
        except Exception:
            pass


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device.upper()}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] No CUDA GPU detected — running on CPU (still works, just slower).")

    print("[INFO] Loading YOLO26n model (first run downloads the weights, ~few MB)...")
    model = YOLO(MODEL_PATH)
    model.to(device)

    conn = init_db(DB_PATH)
    cur = conn.cursor()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check CAMERA_INDEX, or that no other app is using it,")
        print("        or check Windows Settings > Privacy & Security > Camera permissions.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    voice = VoiceAnnouncer(enabled=ENABLE_VOICE, cooldown=VOICE_COOLDOWN_SEC)

    session_counts = defaultdict(int)
    prev_time = time.time()
    snapshot_id = 0

    print("[INFO] Live detection running. Press 'q' to quit, 's' to save a snapshot.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed, stopping.")
            break

        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)
        result = results[0]
        annotated = result.plot()  # draws boxes + labels + confidence automatically

        now_ts = datetime.now().isoformat(timespec="seconds")
        frame_classes = []

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                frame_classes.append(cls_name)
                session_counts[cls_name] += 1
                cur.execute(
                    "INSERT INTO detections (timestamp, class_name, confidence) VALUES (?, ?, ?)",
                    (now_ts, cls_name, conf),
                )
        conn.commit()

        for cls_name in set(frame_classes):
            voice.announce(cls_name)

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        cv2.putText(annotated, f"FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Objects in frame: {len(frame_classes)}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Device: {device.upper()}", (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Object Recognition - YOLO26n", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snapshot_id += 1
            path = os.path.join(SNAPSHOT_DIR, f"snapshot_{snapshot_id}_{int(time.time())}.jpg")
            cv2.imwrite(path, annotated)
            print(f"[INFO] Snapshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    conn.close()

    print("\n========== SESSION SUMMARY ==========")
    if session_counts:
        for cls_name, count in sorted(session_counts.items(), key=lambda x: -x[1]):
            print(f"  {cls_name:<20} {count}")
    else:
        print("  No objects detected.")
    print(f"  Full detection log saved to: {DB_PATH}")
    print(f"  Snapshots (if any) saved to: {SNAPSHOT_DIR}/")
    print("======================================")


if __name__ == "__main__":
    main()
