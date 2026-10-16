# Low-Latency Multi-Camera MJPEG Ingest

**A drop-in Python module that pulls live video from one or more phones over Wi-Fi with minimal latency. It was built as the camera layer for an on-device face-recognition pipeline on the Snapdragon X Elite NPU ([main project](https://github.com/Soorya2201/qualcomm-insightface-npu)).**

## The problem

Live CV demos drift seconds behind reality when frames queue up faster than detection and embedding can consume them. OpenCV's FFMPEG backend also buffers internally, which adds latency you can't fully turn off.

## Design (`frames.py`)

- **Latest frame wins.** A background thread parses the MJPEG byte stream directly (JPEG SOI/EOI markers) and holds exactly one frame, overwriting it. Nothing ever queues, so a slow consumer sees fresh frames, never stale ones.
- **Decode lazily.** The reader stores raw JPEG bytes; decoding happens in `read()` on the consumer's thread, only for frames that are actually used. A 12 fps pipeline on a 30 fps camera decodes 12 images per second, not 30.
- **Pace off the camera.** `read_new()` blocks on a condition variable until a newer frame arrives, so loops don't spin.
- **Many cameras, independent failures.** `CameraBank` runs one reader per phone, so a slow or dead camera can't stall the others. Each reader reconnects automatically.
- **Observable.** `stats()` reports capture fps, consume fps, frames published and frames dropped.
- **Bounded memory.** The parse buffer is capped so a malformed stream can't grow without limit.

## Usage

```python
from frames import MJPEGSource, CameraBank

cam = MJPEGSource("http://<phone-ip>:4747/video", name="phoneA").start()
while True:
    ok, frame = cam.read_new(timeout=1.0)   # newest frame, decoded
    if ok:
        run_detection(frame)

bank = CameraBank(["http://phoneA:4747/video", "http://phoneB:4747/video"]).start()
for name, frame in bank.read_all():
    ...
```

Self-test with live windows: `python frames.py http://<phone-ip>:4747/video`. This works with DroidCam or any MJPEG-over-HTTP source; `opencv_source()` is a fallback that uses OpenCV's FFMPEG backend.

```bash
pip install -r requirements.txt
```
