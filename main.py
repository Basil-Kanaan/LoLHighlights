import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

# Initialize a lock for thread-safe operations on the video capture object
video_capture_lock = threading.Lock()

# Set the desired resolution (1080p)
FRAME_WIDTH = 256
FRAME_HEIGHT = 144

# Time interval (in seconds) between extracted frames
DEFAULT_EXTRACT_INTERVAL = 2

# Threshold value for template matching
MATCH_THRESH = 0.85

# Whether to display frames with matched templates (set from --show-matches)
SHOW_MATCH = False

# Only one match window may be open at a time, since frames are classified on worker threads
display_lock = threading.Lock()

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Template images in grayscale
template_images = {
    "double": cv2.imread(os.path.join(current_dir, 'data', 'images of phrases', 'double.png'), cv2.IMREAD_GRAYSCALE),
    "triple": cv2.imread(os.path.join(current_dir, 'data', 'images of phrases', 'triple.png'), cv2.IMREAD_GRAYSCALE),
    "quadra": cv2.imread(os.path.join(current_dir, 'data', 'images of phrases', 'quadra.png'), cv2.IMREAD_GRAYSCALE),
    "penta": cv2.imread(os.path.join(current_dir, 'data', 'images of phrases', 'penta.png'), cv2.IMREAD_GRAYSCALE),
}


def extract_frames_parallel(video_file_path, seconds_between_frames=DEFAULT_EXTRACT_INTERVAL):
    classified_results = []

    video_capture = cv2.VideoCapture(video_file_path)
    if not video_capture.isOpened():
        print("Error: Failed to open video file.")
        return []
    # else:
    #     #  change the width and height of the capture
    #     video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    #     video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_second = video_capture.get(cv2.CAP_PROP_FPS)
    frames_per_interval = max(1, round(frames_per_second * seconds_between_frames))
    frame_indices = range(0, total_frame_count, frames_per_interval)

    def process_and_classify_frame(video_capture, frame_index, lock):
        with lock:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            frame_retrieved, frame = video_capture.read()

        if frame_retrieved:
            frame_in_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_classification = classify_frame(
                frame_in_grayscale, frame_index / frames_per_second
            )
            if frame_classification:
                classified_results.append(frame_classification)

    with ThreadPoolExecutor() as executor:
        future_to_frame_index = {
            executor.submit(process_and_classify_frame, video_capture, frame_index, video_capture_lock): frame_index for
            frame_index in frame_indices
        }
        for future in as_completed(future_to_frame_index):
            try:
                future.result()
            except Exception as error:
                frame_index = future_to_frame_index[future]
                print(f"Error processing frame {frame_index}: {error}")

    video_capture.release()

    return classified_results


def show_frame_with_border(frame, key):
    """
    Show the frame with a border around the matched template.

    Args:
        frame (numpy.array): Grayscale image frame.
        key (str): Key indicating the matched template.
    """
    result = cv2.matchTemplate(frame, template_images[key], cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= MATCH_THRESH)

    for pt in zip(*loc[::-1]):
        bottom_right = (pt[0] + template_images[key].shape[1], pt[1] + template_images[key].shape[0])
        cv2.rectangle(frame, pt, bottom_right, (0, 0, 255), 2)

    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    cv2.imshow('Frame with Border', frame)
    cv2.waitKey(0)  # Wait until a key is pressed (0 means infinite)
    cv2.destroyAllWindows()


def classify_frame(frame, timestamp):
    """
    Classify a single frame using template matching.

    Args:
        frame (numpy.array): Grayscale image frame.
        timestamp (float): Timestamp of the frame in seconds.

    Returns:
        str: String indicating the matched template and its timestamp.
    """
    for key in template_images:
        result = cv2.matchTemplate(frame, template_images[key], cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= MATCH_THRESH)

        # Output True if there is a match, False otherwise
        if len(loc[0]) > 0:
            if SHOW_MATCH:
                with display_lock:
                    show_frame_with_border(frame.copy(), key)
            formatted_timestamp = format_timestamp(timestamp)
            return f"{key} @ {formatted_timestamp}"
    return None


def format_timestamp(timestamp):
    """
    Format a time float with 2 decimals in seconds to a nicely formatted string in minutes and seconds.

    Args:
        timestamp (float): Time in seconds with 2 decimal places.

    Returns:
        str: Formatted time string (e.g., '1:23' for 1 minute and 23 seconds).
    """
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    return f"{minutes}:{seconds:02d}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video template matching")
    parser.add_argument("video_path", nargs="?", help="Path to the input video file")
    parser.add_argument('--show-matches', action='store_true', help='Flag to display frames with matched templates')
    parser.add_argument('--wait', action='store_true', help='Flag to pause terminal after execution')
    parser.add_argument('--extract-interval', type=float, default=DEFAULT_EXTRACT_INTERVAL,
                        help='Time interval (in seconds) between extracted frames')
    parser.add_argument('--time-elapsed', action='store_true', help='Flag to display elapsed time')
    args = parser.parse_args()

    if not args.video_path:
        args.video_path = input("Copy & Paste the path to your video: ")

    # Path to the input video file
    VID_PATH = args.video_path

    # Flag to display frames with matched templates
    SHOW_MATCH = args.show_matches

    # Start the timer if the --time-elapsed flag is set
    start_time = time.time()

    results = extract_frames_parallel(VID_PATH, seconds_between_frames=args.extract_interval)

    # Sort the results based on the timestamp
    sorted_results = sorted(results, key=lambda x: int(x.split('@')[1].strip().replace(':', '')))

    # Print the sorted matched templates and their timestamps
    print("\n".join(sorted_results))

    # End the timer if the --time-elapsed flag is set and display the elapsed time
    if args.time_elapsed:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    if args.wait:
        input("Press enter to exit...")
