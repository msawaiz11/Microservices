# import pika
# import json
# import time  # For waiting

# def callback(ch, method, properties, body):
#     """Callback function to process received messages."""
#     message = json.loads(body)
    
#     queue_name = message.get("queue", "default_queue")  # Get queue name from message
#     print(f"Using queue: {queue_name}")
#     text = message.get("text", "")
#     status = message.get("status", "")

#     print(f"Received message from {queue_name}: Text: {text}, Status: {status}")

#     ch.basic_ack(delivery_tag=method.delivery_tag)

# # Set up RabbitMQ connection
# RABBITMQ_SERVER = "127.0.0.1"  
# USERNAME = "guest"
# PASSWORD = "guest"

# credentials = pika.PlainCredentials(USERNAME, PASSWORD)
# connection = pika.BlockingConnection(
#     pika.ConnectionParameters(host=RABBITMQ_SERVER, credentials=credentials)
# )
# channel = connection.channel()

# # Wait for a message to determine the queue dynamically
# print("Waiting for messages to determine the queue name...")

# queue_name = None

# # Continuously check for messages instead of exiting
# while queue_name is None:
#     method_frame, header_frame, body = channel.basic_get(queue="audio_processing", auto_ack=False)

#     if body:
#         message = json.loads(body)
#         queue_name = message.get("queue")  # Extract queue name from message
#         print(f"Using queue: {queue_name}")

#         # Declare the extracted queue
#         channel.queue_declare(queue=queue_name)

#         # Start consuming from the dynamic queue
#         channel.basic_consume(queue=queue_name, on_message_callback=callback)
#         channel.start_consuming()
#     else:
#         print("No message received yet. Retrying in 5 seconds...")
#         time.sleep(5)  # Wait before checking again







import pika
import json
import time

def process_text_message(message):
    """Handles text messages"""
    text = message.get("text", "No text provided")
    print(f"📜 Received Text: {text}")

def process_audio_message(message):
    """Handles audio messages"""
    audio_url = message.get("audio_url", "No audio URL provided")
    print(f"🎵 Received Audio File: {audio_url}")

def callback(ch, method, properties, body):
    """Callback function to process received messages."""
    message = json.loads(body)

    queue_name = message.get("queue", "default_queue")  
    message_type = message.get("type", "unknown")  # Determine if it's text or audio

    print(f"\n📥 New Message in Queue: {queue_name}")

    if message_type == "text":
        process_text_message(message)
    elif message_type == "audio":
        process_audio_message(message)
    else:
        print("⚠️ Unknown message type received!")

    ch.basic_ack(delivery_tag=method.delivery_tag)

# Set up RabbitMQ connection
RABBITMQ_SERVER = "127.0.0.1"  
USERNAME = "guest"
PASSWORD = "guest"

credentials = pika.PlainCredentials(USERNAME, PASSWORD)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=RABBITMQ_SERVER, credentials=credentials)
)
channel = connection.channel()

print("🔄 Waiting for messages to determine the queue name...")

queue_name = None

while queue_name is None:
    method_frame, header_frame, body = channel.basic_get(queue="audio_processing", auto_ack=False)

    if body:
        message = json.loads(body)
        queue_name = message.get("queue")
        print(f"✅ Using queue: {queue_name}")

        # Declare the extracted queue
        channel.queue_declare(queue=queue_name)

        # Start consuming from the dynamic queue
        channel.basic_consume(queue=queue_name, on_message_callback=callback)
        channel.start_consuming()
    else:
        print("⏳ No message received yet. Retrying in 5 seconds...")
        time.sleep(5)  












# import pika

# # RabbitMQ server details
# RABBITMQ_SERVER = "192.168.18.32"  # IP of the RabbitMQ server
# RABBITMQ_USER = "myuser"           # Username for RabbitMQ
# RABBITMQ_PASS = "mypassword"       # Password for RabbitMQ

# # Authentication setup
# credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)

# # Establish connection
# connection = pika.BlockingConnection(
#     pika.ConnectionParameters(host=RABBITMQ_SERVER, credentials=credentials)
# )
# channel = connection.channel()

# # Declare the queue (it must match the producer's queue declaration)
# channel.queue_declare(queue='audio_file_send')

# # Callback to handle the received message (audio file data)
# def callback(ch, method, properties, body):
#     print("Received audio file data")
    
#     # Extract filename from properties (optional, if included by producer)
#     filename = "received_audio.mp3"  # Default filename, can be customized
    
#     # Save the binary data to a file
#     with open(filename, 'wb') as f:
#         f.write(body)
    
#     print(f"Audio file saved as {filename}")

# # Set up the consumer to listen for messages on the queue
# channel.basic_consume(queue='audio_file_send', on_message_callback=callback, auto_ack=True)

# print("Waiting for audio files...")
# channel.start_consuming()
