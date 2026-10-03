"""
Time main.py and main_optimized.py on the same video with default settings.

Each run is a full process (interpreter startup and imports included), timed with
time.perf_counter(). The scripts are run alternately so drift affects both equally.

Usage: python benchmark/bench.py synth.mp4 [runs]
"""
import os
import statistics
import subprocess
import sys
import time

repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SCRIPTS = ["main.py", "main_optimized.py"]


def main(video_path, runs):
    times = {script: [] for script in SCRIPTS}
    for _ in range(runs):
        for script in SCRIPTS:
            start = time.perf_counter()
            subprocess.run([sys.executable, os.path.join(repo_dir, script), video_path], check=True,
                           capture_output=True)
            times[script].append(time.perf_counter() - start)

    for script, values in times.items():
        print(f"{script}: {', '.join(f'{v:.1f}' for v in values)} s, median {statistics.median(values):.1f} s")
    ratio = statistics.median(times["main.py"]) / statistics.median(times["main_optimized.py"])
    print(f"Ratio of medians: {ratio:.2f}x")


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
