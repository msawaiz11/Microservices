# from PIL import Image
# from realesrgan_ncnn_py import Realesrgan

# realesrgan = Realesrgan(gpuid=0,model=0)
# with Image.open(r"E:\P_M_services\Real-ESRGAN\inputs\0030.jpg") as image:
#     image = realesrgan.process_pil(image)
#     image.save("output.jpg", quality=95)




# import cv2
# import os
# from PIL import Image
# from realesrgan_ncnn_py import Realesrgan
# import numpy as np
# # Initialize Real-ESRGAN
# realesrgan = Realesrgan(gpuid=0, model=0)

# # Input and output video paths

# input_video_path = r"E:\P_M_services\Real-ESRGAN\inputs\london2.mp4"
# output_video_path = r"output_video.mp4"

# # Create a directory to store frames
# frame_dir = "frames"
# os.makedirs(frame_dir, exist_ok=True)

# # Open video file
# cap = cv2.VideoCapture(input_video_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS of input video
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# frame_count = 0

# # Process each frame
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break  # Break when video ends

#     # Convert frame (BGR -> RGB) for PIL
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_image = Image.fromarray(frame_rgb)

#     # Enhance using Real-ESRGAN
#     enhanced_image = realesrgan.process_pil(pil_image)

#     # Convert back to OpenCV format (RGB -> BGR)
#     enhanced_frame = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

#     # Save frame as image
#     frame_path = os.path.join(frame_dir, f"frame_{frame_count:04d}.png")
#     cv2.imwrite(frame_path, enhanced_frame)
    
#     frame_count += 1

# cap.release()
# print(f"Processed {frame_count} frames.")

# # Reconstruct video from enhanced frames
# frame_files = sorted(os.listdir(frame_dir))  # Sort frames by name
# frame_example = cv2.imread(os.path.join(frame_dir, frame_files[0]))  # Read first frame

# # Define video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_example.shape[1], frame_example.shape[0]))

# # Write enhanced frames to output video
# for frame_file in frame_files:
#     frame = cv2.imread(os.path.join(frame_dir, frame_file))
#     out.write(frame)

# out.release()
# print(f"Enhanced video saved at: {output_video_path}")
























import cv2
import numpy as np
from PIL import Image
from realesrgan_ncnn_py import Realesrgan

# Initialize Real-ESRGAN model
realesrgan = Realesrgan(gpuid=0, model=0)

# Input and output video paths
input_video_path = r"E:\P_M_services\Real-ESRGAN\inputs\london2.mp4"
output_video_path = r"enhanced_video.mp4"

# Define target resolution or scale factor
target_resolution = (1920, 1080)  # Set to (width, height) or "2x"

# Read input video
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video FPS
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if video_width == 0 or video_height == 0:
    print("\n❌ Error: Unable to read video file! Check if the file is corrupted.\n")
    exit(1)

# Determine scaling factor
aspect_ratio = video_width / video_height

if isinstance(target_resolution, str):  # Handles "2x", "3x" scaling cases
    scale_factor = int(target_resolution[0])
    final_width = video_width * scale_factor
    final_height = video_height * scale_factor
else:
    final_width, final_height = target_resolution
    scale_factor = max(final_width / video_width, final_height / video_height)

# Ensure final resolution is even
while (int(video_width * scale_factor) % 2 != 0) or (int(video_height * scale_factor) % 2 != 0):
    scale_factor += 0.01

# Final adjusted width and height
final_width = int(video_width * scale_factor)
final_height = int(video_height * scale_factor)

# Define video writer for output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (final_width, final_height))

print(f"Processing video frames... Upscaling to {final_width}x{final_height}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Convert frame to PIL image for Real-ESRGAN processing
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Enhance image with Real-ESRGAN
    enhanced_image = realesrgan.process_pil(image)

    # Convert back to OpenCV format
    enhanced_frame = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

    # Resize frame to match final resolution
    enhanced_frame = cv2.resize(enhanced_frame, (final_width, final_height), interpolation=cv2.INTER_CUBIC)

    # Write frame directly to output video
    out.write(enhanced_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Enhanced video saved at: {output_video_path}\n")
