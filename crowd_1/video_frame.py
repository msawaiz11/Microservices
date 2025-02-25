import cv2
import os

def save_frame(frame, frame_number, output_dir):
    """Save the current frame as an image file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"frame_{frame_number:04d}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")

def is_scene_completely_different_histogram(frame1, frame2, threshold=0.5):
    """
    Compare two frames using histogram comparison.
    Returns True if the frames are significantly different.
    """
    # Convert frames to grayscale
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    hist1 = cv2.calcHist([gray_frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray_frame2], [0], None, [256], [0, 256])

    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1)
    hist2 = cv2.normalize(hist2, hist2)

    # Compare histograms using correlation
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    print(f"Histogram similarity score: {score}")

    # If similarity score is below threshold, consider it a new scene
    return score < threshold

def process_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Cannot read the first frame.")
        return

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, current_frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Compare with the previous frame
        if is_scene_completely_different_histogram(prev_frame, current_frame, threshold=0.9):  # Adjust threshold if needed
            saved_frame_count += 1
            save_frame(current_frame, saved_frame_count, output_dir)
            prev_frame = current_frame

    cap.release()
    print(f"Processing complete. {saved_frame_count} frames saved.")

if __name__ == "__main__":
    # Input video path
    video_file = r"D:\user dataset\crowd-data-24-IK-protest\Psc Tool plaza\24-11-2024\20241124135519760.mp4"
    # Output directory to save frames
    output_directory = "output_frames"
    process_video(video_file, output_directory)
