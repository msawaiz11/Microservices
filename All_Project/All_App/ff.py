import os
from pathlib import Path

# Get the absolute path of the directory where the script is running
current_working_dir = Path(os.getcwd())  # Get current working directory

# Set the parent directory outside the project folder
parent_dir = current_working_dir.parent.parent  # Go up two levels from the current working directory

# Define the relative path from the parent directory
relative_path = 'shared_storage/Converted_videos'

# Combine the parent directory with the relative path
shared_storage_dir = parent_dir / relative_path

# Create the directory if it doesn't exist
shared_storage_dir.mkdir(parents=True, exist_ok=True)

# Define the output filename and output path
output_filename = "test.mp3"
output_path = shared_storage_dir / output_filename 

# Continue your FFmpeg setup...
