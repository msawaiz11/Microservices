import os
from pathlib import Path

# Relative path (could be from the current directory)
relative_path = 'shared_storage/Converted_videos'
project_dir = Path(__file__).resolve().parent


print('prokect', project_dir)

# Get the absolute path
absolute_path = os.path.abspath(relative_path)

print("Absolute Path:", absolute_path)
