import os
model_path = os.path.abspath(r"All_Project\weights\x_net.pth")
print(model_path)  # Debugging: print the absolute path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"File not found: {model_path}")


import os

directory = os.path.abspath("All_Project/weights")
print(f"Checking files in: {directory}")

if os.path.exists(directory):
    print("Files found:", os.listdir(directory))
else:
    print("Directory not found!")
