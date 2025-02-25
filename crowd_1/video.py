import cv2
import os

# Set the video path
video_path = r"D:\user dataset\crowd-data-24-IK-protest\Psc Tool plaza\24-11-2024\20241124135519760.mp4"
cap = cv2.VideoCapture(video_path)

# Set up the folder to save screenshots
output_folder = "screenshots"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Define rectangle parameters
ret, frame = cap.read()
if not ret:
    print("Error reading video.")
    cap.release()
    exit()

height, width = frame.shape[:2]

# Rectangle size and initial position
rect_height = 100  # Height of the rectangle
rect_y_position = height - rect_height  # Start from the bottom

# Initialize counters
saved_frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Define the rectangle position
    rect_start = (0, rect_y_position)
    rect_end = (width, rect_y_position + rect_height)

    # Draw the rectangle on the current frame
    cv2.rectangle(frame, rect_start, rect_end, (0, 0, 255), 2)

    # Check if the top of the rectangle reaches the top of the frame
    if rect_y_position <= 0:
        # Save the frame when the rectangle reaches the top of the video
        frame_filename = os.path.join(output_folder, f"saved_frame_{saved_frame_counter}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        saved_frame_counter += 1

        # Reset the rectangle position to continue tracking
        rect_y_position = height - rect_height

    # Move the rectangle up for the next frame
    rect_y_position -= 8  # Adjust the speed of movement here if necessary

    # Optional: Display the frame with the rectangle for preview
    cv2.imshow("Frame with Rectangle", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
