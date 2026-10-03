import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from decord import VideoReader, cpu

# Initialize a lock for thread-safe operations on the video capture object
video_capture_lock = threading.Lock()

# Set the desired resolution
FRAME_WIDTH = 854  # 256
FRAME_HEIGHT = 480  # 144

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


def load_and_resize_image(image_path, scale_percent=50):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Calculate the new dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # Resize the image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


template_images = {
    "double": load_and_resize_image(os.path.join(current_dir, 'data', 'images of phrases', 'double.png'), 100),
    "triple": load_and_resize_image(os.path.join(current_dir, 'data', 'images of phrases', 'triple.png'), 100),
    "quadra": load_and_resize_image(os.path.join(current_dir, 'data', 'images of phrases', 'quadra.png'), 100),
    "penta": load_and_resize_image(os.path.join(current_dir, 'data', 'images of phrases', 'penta.png'), 100),
}


def extract_frames(video_file_path, seconds_between_frames):
    """
    Extract frames from the specified video file at the given interval of seconds.

    Args:
       video_file_path (str): Path to the video file.
       seconds_between_frames (float): Interval between frames to extract in seconds.

    Returns:
       list: A list of classified frames.
    """
    vr = VideoReader(video_file_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    frames_to_extract = max(1, round(fps * seconds_between_frames))

    total_frames = len(vr)
    frame_indices = list(range(0, total_frames, frames_to_extract))

    # Use ThreadPoolExecutor to process frames in parallel
    with ThreadPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(classify_frame, cv2.cvtColor(vr[idx].asnumpy(), cv2.COLOR_RGB2GRAY), idx / fps): idx for idx
            in frame_indices}
        results = (future.result() for future in as_completed(future_to_idx))
        extracted_frames = [result for result in results if result]

    return extracted_frames


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
    Convert a timestamp in seconds to a formatted string in minutes and seconds.

    Args:
        timestamp (float): The timestamp in seconds.

    Returns:
        str: The formatted timestamp as a string in 'minutes:seconds' format.
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

    results = extract_frames(VID_PATH, seconds_between_frames=args.extract_interval)

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
