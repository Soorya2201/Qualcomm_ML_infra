"""
frames.py
Low latency frame ingest from one or more DroidCam phones over WiFi.

Design rule: latest frame wins. The reader thread never queues. It holds exactly
one JPEG at a time and overwrites it. If the consumer (detection, embedding) is
slower than the camera, old frames are discarded instead of piling up. Queuing
here is what makes a live demo drift several seconds behind reality.

Second design rule: the reader thread stores raw JPEG bytes and does not decode.
Decoding happens in read(), on the consumer's thread, only for frames that are
actually used. If your pipeline runs at 12 fps off a 30 fps camera, you decode
12 images per second instead of 30.

Usage:
    from frames import MJPEGSource

    cam = MJPEGSource("http://10.36.254.185:4747/video", name="phoneA").start()
    while True:
        ok, frame = cam.read()
        if not ok:
            continue
        ...

    cam.stop()

Requires: pip install opencv-python numpy requests
"""

import threading
import time
import cv2
import numpy as np
import requests


SOI = b"\xff\xd8"   # JPEG start of image
EOI = b"\xff\xd9"   # JPEG end of image


class MJPEGSource:
    """Pulls an MJPEG HTTP stream in a background thread, keeps only the newest frame."""

    def __init__(self, url, name=None, target_fps=30.0, timeout=5.0,
                 reconnect_delay=1.0, max_buffer=4 * 1024 * 1024):
        self.url = url
        self.name = name or url
        self.target_fps = float(target_fps)
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.max_buffer = max_buffer

        self._jpeg = None
        self._seq = 0
        self._last_seq_read = 0   # frames are numbered from 1; 0 means "none read yet"
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread = None

        self._connected = False
        self._recv_times = []
        self._decoded = 0
        self._dropped = 0
        self._started_at = None

    # ---------------------------------------------------------------- lifecycle

    def start(self):
        if self._thread is not None:
            return self
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, name=f"mjpeg-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------- reader thread

    def _run(self):
        min_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        while not self._stop.is_set():
            try:
                with requests.get(self.url, stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    self._connected = True
                    buf = bytearray()
                    last_publish = 0.0

                    for chunk in r.iter_content(chunk_size=8192):
                        if self._stop.is_set():
                            break
                        if not chunk:
                            continue
                        buf.extend(chunk)

                        # Guard against a stream that never yields a valid frame.
                        if len(buf) > self.max_buffer:
                            del buf[:-65536]

                        # Pull every complete JPEG out of the buffer, keep the last one.
                        newest = None
                        while True:
                            i = buf.find(SOI)
                            if i < 0:
                                break
                            j = buf.find(EOI, i + 2)
                            if j < 0:
                                if i > 0:
                                    del buf[:i]
                                break
                            newest = bytes(buf[i:j + 2])
                            del buf[:j + 2]

                        if newest is None:
                            continue

                        now = time.time()
                        if min_interval and (now - last_publish) < min_interval:
                            # Source is faster than we want. Skip without storing.
                            self._dropped += 1
                            continue
                        last_publish = now
                        self._publish(newest, now)

            except Exception:
                self._connected = False
                if self._stop.is_set():
                    break
                time.sleep(self.reconnect_delay)

        self._connected = False

    def _publish(self, jpeg_bytes, ts):
        with self._new_frame:
            if self._jpeg is not None and self._seq > self._last_seq_read:
                self._dropped += 1        # consumer never took the previous one
            self._jpeg = jpeg_bytes
            self._seq += 1
            self._recv_times.append(ts)
            if len(self._recv_times) > 60:
                self._recv_times = self._recv_times[-60:]
            self._new_frame.notify_all()

    # ------------------------------------------------------------------ consumer

    def read(self):
        """Non blocking. Returns (ok, frame). Decodes the newest JPEG available.
        Returns the same frame again if no new one has arrived."""
        with self._lock:
            jpeg, seq = self._jpeg, self._seq
        if jpeg is None:
            return False, None
        self._last_seq_read = seq
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False, None
        self._decoded += 1
        return True, frame

    def read_new(self, timeout=1.0):
        """Blocks until a frame newer than the last one read arrives.
        Use this to pace your loop off the camera instead of spinning."""
        deadline = time.time() + timeout
        with self._new_frame:
            while self._seq <= self._last_seq_read:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False, None
                self._new_frame.wait(remaining)
            jpeg, seq = self._jpeg, self._seq
        self._last_seq_read = seq
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return False, None
        self._decoded += 1
        return True, frame

    # --------------------------------------------------------------------- stats

    @property
    def connected(self):
        return self._connected

    def stats(self):
        with self._lock:
            times = list(self._recv_times)
        if len(times) >= 2:
            span = times[-1] - times[0]
            capture_fps = (len(times) - 1) / span if span > 0 else 0.0
        else:
            capture_fps = 0.0
        elapsed = max(1e-6, time.time() - (self._started_at or time.time()))
        return {
            "name": self.name,
            "connected": self._connected,
            "capture_fps": round(capture_fps, 1),
            "consume_fps": round(self._decoded / elapsed, 1),
            "frames_published": self._seq,
            "frames_dropped": self._dropped,
        }


class CameraBank:
    """Several phones as one object. Keeps them independent so a slow or dead
    phone cannot stall the others."""

    def __init__(self, urls, target_fps=30.0):
        self.sources = []
        for i, u in enumerate(urls):
            name = f"cam{i}"
            if isinstance(u, (tuple, list)):
                u, name = u
            self.sources.append(MJPEGSource(u, name=name, target_fps=target_fps))

    def start(self):
        for s in self.sources:
            s.start()
        return self

    def stop(self):
        for s in self.sources:
            s.stop()

    def read_all(self):
        """Returns list of (name, frame) for every camera that currently has one."""
        out = []
        for s in self.sources:
            ok, f = s.read()
            if ok:
                out.append((s.name, f))
        return out

    def stats(self):
        return [s.stats() for s in self.sources]


# ------------------------------------------------------------------------ opencv
def opencv_source(url):
    """Fallback if the raw parser misbehaves on your phone. Slightly higher
    latency because the FFMPEG backend buffers internally."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# -------------------------------------------------------------------- self test
if __name__ == "__main__":
    import sys

    urls = sys.argv[1:] or ["http://10.36.254.185:4747/video"]
    bank = CameraBank(urls, target_fps=30.0).start()
    print("connecting, press esc in the window to quit")

    last_print = 0.0
    try:
        while True:
            frames = bank.read_all()
            for name, frame in frames:
                cv2.imshow(name, frame)

            now = time.time()
            if now - last_print > 1.0:
                last_print = now
                for s in bank.stats():
                    print(f"{s['name']:6s} connected={s['connected']} "
                          f"capture={s['capture_fps']:5.1f} fps  "
                          f"consume={s['consume_fps']:5.1f} fps  "
                          f"published={s['frames_published']}  dropped={s['frames_dropped']}")

            if cv2.waitKey(1) == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        bank.stop()
        cv2.destroyAllWindows()
