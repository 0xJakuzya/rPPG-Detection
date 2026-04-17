import threading
from queue import Queue, Empty

import cv2
import numpy as np

import config


class VideoCapture:
    def __init__(self):
        self.cam = cv2.VideoCapture(config.CAMERA_INDEX)
        self.fps = config.FPS_TARGET
        self.running = False
        self.frame_queue = Queue(maxsize=3)
        self.latest_frame = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

    def capture_loop(self):
        while self.running:
            ret, frame = self.cam.read()
            if not ret:
                self.running = False
                break
            self.latest_frame = frame
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            try:
                self.frame_queue.put_nowait(frame)
            except Exception:
                pass

    def read(self):
        return self.latest_frame

    def get_frame(self, timeout=1.0):
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self):
        self.running = False
        self.cam.release()