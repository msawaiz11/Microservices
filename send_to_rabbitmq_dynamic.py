





# import pika
# import json

# # RabbitMQ server details

# RABBITMQ_SERVER = "192.168.18.32"  # Change to "localhost" if running on the same PC
# RABBITMQ_USER = "myuser"       # Use the same user created in the consumer
# RABBITMQ_PASS = "mypassword"   # Use the same password

# # Authentication setup
# credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)

# # Establish connection
# connection = pika.BlockingConnection(
#     pika.ConnectionParameters(host=RABBITMQ_SERVER, credentials=credentials)
# )
# channel = connection.channel()

# # Declare the queue (must match the consumer queue)
# channel.queue_declare(queue='audio_processing')

# # Create a message
# message = json.dumps({"text": "Hello from another user!", "status": "sent"})

# # Publish the message to the queue
# channel.basic_publish(exchange='', routing_key='audio_processing', body=message)

# print("Message sent to RabbitMQ!")

# # Close the connection
# connection.close()




# import pika
# import json

# RABBITMQ_SERVER = "127.0.0.1"  # Use localhost for testing on the same PC
# QUEUE_NAME = "audio_processing"  # Change this dynamically as needed

# # Connect to RabbitMQ
# connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_SERVER))
# channel = connection.channel()

# # Declare the queue dynamically
# channel.queue_declare(queue=QUEUE_NAME)

# # Send a test message
# message = json.dumps({
#     "queue": QUEUE_NAME,  # Send the queue name as part of the message
#     "text": "Test message from producer!",
#     "status": "sent"
# })

# channel.basic_publish(exchange='', routing_key=QUEUE_NAME, body=message)

# print(f"Message sent to RabbitMQ on queue: {QUEUE_NAME}")

# # Close the connection
# connection.close()



import pika
import json

RABBITMQ_SERVER = "127.0.0.1"  
QUEUE_NAME = "audio_processing"  

# Connect to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_SERVER))
channel = connection.channel()

# Declare the queue dynamically
channel.queue_declare(queue=QUEUE_NAME)

# Example: Sending a text message
text_message = json.dumps({
    "queue": QUEUE_NAME,
    "type": "text",  # Identify it as text
    "text": "This is a test message",
    "status": "sent"
})

# Example: Sending an audio message
audio_message = json.dumps({
    "queue": QUEUE_NAME,
    "type": "audio",  # Identify it as audio
    "audio_url": "https://example.com/audio.mp3",  # Simulating an audio file link
    "status": "sent"
})

# Choose which type to send
message_to_send = audio_message  # Change this to `audio_message` for testing audio

channel.basic_publish(exchange='', routing_key=QUEUE_NAME, body=message_to_send)

print(f"Message sent to RabbitMQ on queue: {QUEUE_NAME}")

# Close the connection
connection.close()
