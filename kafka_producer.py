from kafka import KafkaProducer
import os
from env import KAFKA_BROKER, KAFKA_TOPIC_NAME, TRAINING_DATA_PATH

producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER)

# Read CSV and send data to Kafka

for file_path in os.listdir(TRAINING_DATA_PATH):
    with open(TRAINING_DATA_PATH+file_path, 'r') as file:
        for line in file:
            producer.send(KAFKA_TOPIC_NAME, value=line.strip().encode('utf-8'))

producer.close()
