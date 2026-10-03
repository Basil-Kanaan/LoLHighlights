"""
Generate a synthetic test clip: scrolling random noise with each multi-kill banner
template pasted in at its native size for 3 seconds at a known time.

Requires ffmpeg on PATH (used to encode H.264).

Usage: python benchmark/make_synthetic_video.py synth.mp4 [duration_seconds]
"""
import os
import subprocess
import sys

import cv2
import numpy as np

WIDTH, HEIGHT, FPS = 1920, 1080, 30
BANNER_SECONDS = 3

# Second at which each banner appears (1:00, 2:30, 5:00, 7:30)
EVENTS = {"double": 60, "triple": 150, "quadra": 300, "penta": 450}

template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'images of phrases')


def main(out_path, duration):
    templates = {key: cv2.imread(os.path.join(template_dir, f'{key}.png')) for key in EVENTS}
    rng = np.random.default_rng(0)
    background = cv2.resize(rng.integers(0, 255, (HEIGHT // 8, WIDTH // 8, 3), dtype=np.uint8), (WIDTH, HEIGHT))

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE)

    for i in range(FPS * duration):
        t = i / FPS
        frame = np.roll(background, i * 4, axis=1)
        for key, start in EVENTS.items():
            if start <= t < start + BANNER_SECONDS:
                banner = templates[key]
                y, x = 200, (WIDTH - banner.shape[1]) // 2
                frame[y:y + banner.shape[0], x:x + banner.shape[1]] = banner
        ffmpeg.stdin.write(frame.tobytes())

    ffmpeg.stdin.close()
    ffmpeg.wait()


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 600)
