





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





import pika
import json

RABBITMQ_SERVER = "127.0.0.1"  # Use localhost for testing on the same PC

# Connect to RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_SERVER))
channel = connection.channel()

# Declare the queue
channel.queue_declare(queue='audio_processing')

# Send a test message
message = json.dumps({"text": "Test message from producer!", "status": "sent"})
channel.basic_publish(exchange='', routing_key='audio_processing', body=message)

print("Message sent to RabbitMQ!")

# Close the connection
connection.close()


