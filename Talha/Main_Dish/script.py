import subprocess
import time

# Run the first command
print("Starting Daphne server...")
daphne_process = subprocess.Popen(["daphne","Main_Dish.asgi:application"])

# Wait for 2 seconds
time.sleep(2)

# Run the second command
print("Starting Celery worker...")
celery_process = subprocess.Popen(["celery", "-A", "Main_Dish", "worker", "-l", "info", "-P", "gevent"])

# Keep the script running to ensure the processes stay alive
try:
    daphne_process.wait()
    celery_process.wait()
except KeyboardInterrupt:
    print("Terminating processes...")
    daphne_process.terminate()
    celery_process.terminate()
