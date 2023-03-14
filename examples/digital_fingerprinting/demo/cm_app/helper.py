from confluent_kafka import Producer
import json
import logging

logging.basicConfig()
logger = logging.getLogger("logger")

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))


def publish_message(message):
    p = Producer({'bootstrap.servers': 'localhost:9092'})

    p.poll(0)

    # Asynchronously produce a message. The delivery report callback will
    # be triggered from the call to poll() above, or flush() below, when the
    # message has been successfully delivered or failed permanently.
    p.produce('test_cm', message.encode('utf-8'))

    # Wait for any outstanding messages to be delivered and delivery report
    # callbacks to be triggered.
    p.flush()

def process_cm(request):
    control_messages_json = request.form.get("control-messages-json")
    publish_message(control_messages_json)
    logging.error(control_messages_json)
    data = {
        "status": "Successfully published task to kafka topic.",
        "status_code": 200,
        "control_messages": json.loads(control_messages_json)
    }
    data = json.dumps(data, indent=4)
    return data