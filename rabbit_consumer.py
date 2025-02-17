import pika
import json

def callback(ch, method, properties, body):
    """Callback function to process received messages."""
    message = json.loads(body)
    text = message.get("text", "")
    status = message.get("status", "")

    print(f"Received message from RabbitMQ: Text: {text}, Status: {status}")


    ch.basic_ack(delivery_tag=method.delivery_tag)

# Set up RabbitMQ connection


# RABBITMQ_SERVER = "192.168.18.32"  # Replace with your server's IP
# USERNAME = "myuser"
# PASSWORD = "mypassword"
    
RABBITMQ_SERVER = "127.0.0.1"  # Replace with your server's IP
USERNAME = "guest"
PASSWORD = "guest"

credentials = pika.PlainCredentials(USERNAME, PASSWORD)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=RABBITMQ_SERVER, credentials=credentials)
)

# connection = pika.BlockingConnection(pika.ConnectionParameters(host='127.0.0.1'))
channel = connection.channel()



# Declare the same queue to receive messages
channel.queue_declare(queue='audio_processing')

# Start consuming messages
channel.basic_consume(queue='audio_processing', on_message_callback=callback)

print("Waiting for messages from RabbitMQ. To exit, press CTRL+C")
channel.start_consuming()











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
