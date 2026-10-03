# LoLHighlights

A command-line tool that scans a League of Legends gameplay recording and prints the timestamps of multi-kills (double, triple, quadra and penta kills), so you can jump straight to the highlights instead of scrubbing through a full match.

```
$ python main_optimized.py synth.mp4 --time-elapsed
double @ 1:00
double @ 1:02
triple @ 2:30
triple @ 2:32
quadra @ 5:00
quadra @ 5:02
penta @ 7:30
penta @ 7:32
Elapsed Time: 15.98 seconds
```

(Output from the synthetic test clip described under [Measured speed difference](#measured-speed-difference).)

## How it works

When a player gets a multi-kill, the game draws a banner on screen ("DOUBLE KILL!", "PENTAKILL!", and so on). Instead of training a model, the tool looks for those banners directly:

1. **Sample frames.** It reads one frame every `--extract-interval` seconds (default 2) rather than decoding every frame.
2. **Convert to grayscale.** Each sampled frame is converted to a single channel to match the templates.
3. **Template-match.** Each frame is compared against four reference crops in [`data/images of phrases/`](data/images%20of%20phrases) using OpenCV's `cv2.matchTemplate` with normalized cross-correlation (`TM_CCOEFF_NORMED`). Any location scoring at or above `MATCH_THRESH = 0.85` counts as a hit.
4. **Report.** Hits are labelled with the template name and the frame's timestamp, sorted by time and printed as `<type> @ m:ss`.

Frames are classified concurrently with a `ThreadPoolExecutor`. OpenCV releases the GIL inside `matchTemplate`, so the matching runs in parallel across threads.

## `main.py` vs `main_optimized.py`

Both scripts take the same arguments, use the same templates and threshold, and give the same output. The difference is how frames are read from the video.

| | `main.py` | `main_optimized.py` |
|---|---|---|
| Video decoding | `cv2.VideoCapture` | [decord](https://github.com/dmlc/decord) `VideoReader` |
| Frame access | Each worker thread takes a lock, seeks with `CAP_PROP_POS_FRAMES`, and reads one frame. Decoding is serialized; only matching runs in parallel. | The main thread decodes frames by index (`vr[idx]`) while submitting them to the pool. Workers only run classification. |
| Packaged by `LoLHighlights.spec` | Yes | No |

### Measured speed difference

On the test below, `main_optimized.py` was about 1.9x faster than `main.py`. I did not profile the scripts, so I can't say how much of that comes from decord's decoding and how much from the different threading layout.

| Script | Wall time, 5 runs (s) | Median (s) |
|---|---|---|
| `main.py` | 31.6, 33.8, 32.3, 33.2, 31.0 | 32.3 |
| `main_optimized.py` | 17.7, 17.2, 17.6, 16.0, 16.0 | 17.2 |

Ratio of medians: **1.87x**.

How this was measured:

- **Input:** a synthetic 10-minute clip, 1920x1080, 30 fps, H.264 (libx264, yuv420p) encoded with ffmpeg, 18,000 frames. The background is scrolling random noise, and each of the four banner templates is pasted in at its native size for 3 seconds at 1:00, 2:30, 5:00 and 7:30. It is **not** real gameplay footage.
- **Command:** each script was run with its default settings (`python <script> synth.mp4`, 2-second interval). The full process wall time was measured with `time.perf_counter()` around `subprocess.run`, so it includes interpreter startup and imports. The two scripts were run alternately, 5 times each.
- **Machine:** Intel Core i7-11700KF (8 cores / 16 threads), Windows 11, Python 3.12.10, opencv-python 5.0.0.93, numpy 2.5.3, decord 0.6.0.

Both scripts found all four banners at the correct timestamps on this clip. Results on other hardware, codecs or resolutions will differ.

To reproduce (needs ffmpeg on `PATH`):

```bash
python benchmark/make_synthetic_video.py synth.mp4
python benchmark/bench.py synth.mp4
```

## Requirements

- Python 3 (tested on 3.12)
- `opencv-python`, `numpy`
- `decord` for `main_optimized.py` only. decord 0.6.0 publishes wheels for Windows x86-64 and Linux x86-64. On macOS it only has Intel wheels for Python 3.6 to 3.8, and there is no Apple Silicon wheel.

## Setup

```bash
git clone https://github.com/Basil-Kanaan/LoLHighlights.git
cd LoLHighlights
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

If you only plan to run `main.py`, `pip install opencv-python numpy` is enough.

## Usage

```
python main_optimized.py [video_path] [--extract-interval SECONDS] [--time-elapsed] [--wait] [--show-matches]
```

| Argument | Description |
|---|---|
| `video_path` | Path to the recording. If omitted, the script asks for it interactively. |
| `--extract-interval SECONDS` | Seconds between sampled frames (default `2`). Smaller values catch shorter banners but take longer. |
| `--time-elapsed` | Print total processing time. |
| `--wait` | Wait for Enter before exiting, so the console window stays open when the packaged executable is double-clicked. |
| `--show-matches` | Open a window for each matched frame with the banner outlined. Press any key to close it and continue. |

Examples:

```bash
# Scan a recording with the defaults (one frame every 2 seconds)
python main_optimized.py "C:\Videos\ranked_game.mp4"

# Sample every second for better recall, and print timing
python main.py "C:\Videos\ranked_game.mp4" --extract-interval 1 --time-elapsed

# Check what was matched, one window per hit
python main_optimized.py "C:\Videos\ranked_game.mp4" --show-matches

# No argument: the script asks you to paste a path
python main_optimized.py
```

### Building a Windows executable

`LoLHighlights.spec` packages `main.py` and the template images with PyInstaller:

```bash
pip install pyinstaller
pyinstaller LoLHighlights.spec
```

The executable is written to `dist/LoLHighlights.exe`.

## Known limitations

- **Resolution and HUD scale.** Template matching is not scale-invariant. The templates are fixed-size crops (166 to 200 px wide), so the recording has to match the resolution and in-game UI scale they were cropped from. The repo does not record what that resolution was. Footage at another size will produce no matches.
- **Fixed threshold.** `MATCH_THRESH` is hard-coded at 0.85, and accuracy has not been benchmarked on real recordings. The synthetic test above checks the pipeline end to end with banners pasted in pixel-exact.
- **Duplicate reports.** A banner that stays visible across several sampled frames is reported once per frame. In the test above, every 3-second banner was listed twice (for example `double @ 1:00` and `double @ 1:02`). Hits are not merged into events.
- **Sampling can miss banners.** Only one frame per interval is checked, so a banner shorter than `--extract-interval` can fall between samples. Timestamps mark the sampled frame, not when the banner first appeared, and are truncated to whole seconds.
- **Speed.** Even the faster script decodes every sampled frame at full resolution and matches four templates against the whole frame. Restricting the search to the region where banners appear, or downscaling, would cut the work but has not been done.
- **Packaging.** `LoLHighlights.spec` uses Windows path separators and only packages `main.py`.
