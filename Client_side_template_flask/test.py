
from pathlib import Path
script_dir = Path(__file__).resolve().parent  # Path of the current script

# Get the parent directory two levels above the script's directory
parent_dir = script_dir.parent  # Go up two levels from the script's directory

# Define the relative path to the shared storage directory
relative_path = 'shared_storage/Converted_videos'

# Combine the parent directory with the relative path
shared_storage_dir = parent_dir / relative_path

print('shared_storage dir', shared_storage_dir)
