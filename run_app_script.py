import subprocess
import time

commands = [
    (r"E:\P_M_services\All_Project", "celery -A All_Project worker -l info -P gevent"),
    (r"E:\P_M_services\All_Project", "python manage.py runserver"),
    
    (r"E:\P_M_services\Client_side_template_flask", "python app.py"),
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
