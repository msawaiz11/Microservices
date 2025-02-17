import subprocess
import time

commands = [
    (r"C:\Program Files\kibana\bin", "kibana.bat"),
    (r"C:\Program Files\elasticsearch\bin", "elasticsearch.bat"),
]

processes = []

for path, command in commands:
    print(f"Starting: {command}")
    process = subprocess.Popen(command, cwd=path, shell=True)
    processes.append(process)
    time.sleep(5)  # Give some time before starting the next service

print("All services started!")

try:
    for process in processes:
        process.wait()
except KeyboardInterrupt:
    print("Shutting down services...")
    for process in processes:
        process.terminate()
