import json
import boto3
import time
import logging
import requests
import urllib
from decimal import Decimal
from model import get_model, predict
from llama import llama_generate

logging.basicConfig(filename="main.log", level=logging.DEBUG)
MAX_HISTORY_TURNS = 6

def handle_message(telegram_token, table, predict_func, message):
    logging.info("Handling message")
    logging.info(message)

    decoded_message=json.loads(message['Body'])
    body_json = decoded_message['body-json']
    update_id = body_json['update_id']
    chat_id = body_json['message']['chat']['id']
    first_name = body_json['message']['chat']['first_name']
    last_name = body_json['message']['chat']['last_name']
    username = body_json['message']['chat']['username']
    message_text = body_json['message']['text'].strip()

    if MAX_HISTORY_TURNS > 0:
        # Get the last MAX_HISTORY_TURNS messages from the chat history
        response = table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('chat_id').eq(chat_id),
            ScanIndexForward=False,
            Limit=MAX_HISTORY_TURNS
        )

        history_items = response.get('Items', [])
        history_items.reverse()

    send_typing(telegram_token, chat_id)

    table.put_item(Item={
        "update_id": update_id,
        "chat_id": chat_id,
        "first_name": f"{first_name} {last_name} (@{username})",
        "message": message_text,
        "from": 0,
        "timestamp": Decimal(time.time())
    })
    if message_text == "/start":
        send_message(telegram_token, "Hi! I'm a chatbot. Ask me anything.\nUse /reset to reset the context", chat_id)
    elif message_text == "/reset":
        pass
    else:
        reply = predict_func(first_name, history_items, message_text)

        if reply != "":
            send_message(telegram_token, reply, chat_id)
            
            table.put_item(Item={
                "update_id": update_id,
                "chat_id": chat_id,
                "first_name": "",
                "message": reply,
                "from": 1,
                "timestamp": Decimal(time.time())
            })


def send_typing(api_token, chat_id):
    URL = "https://api.telegram.org/bot{}/".format(api_token)
    url = URL + "sendChatAction?action=typing&chat_id={}".format(chat_id)
    r = requests.get(url)
    logging.info(r.text)


def send_message(api_token, text, chat_id):
    logging.info(text)
    URL = "https://api.telegram.org/bot{}/".format(api_token)
    text = urllib.parse.quote(text)
    url = URL + "sendMessage?text={}&chat_id={}".format(text, chat_id)
    r = requests.get(url)
    logging.info(r.text)


def send_commands(api_token):
    URL = "https://api.telegram.org/bot{}/".format(api_token)
    url = URL + "setMyCommands?commands={}&scope=default".format(json.dumps([
        {
            "command": "/reset",
            "description": "Reset the context"
        },
        {
            "command": "/retry",
            "description": "Regenerate for the last message"
        }
    ]))
    r = requests.get(url)
    logging.info(r.text)


if __name__ == '__main__':
    # Load credentials from JSON file.
    print("Initializing...")

    with open('credentials.json', 'r') as f:
        credentials = json.load(f)
        session = boto3.Session(
            aws_access_key_id=credentials["aws_access_key_id"],
            aws_secret_access_key=credentials["aws_secret_access_key"],
            region_name=credentials["aws_region"]
        )
    
    queue_url = credentials["queue_url"]
    telegram_token = credentials["telegram_bot_token"]

    sqs_client = session.client('sqs')
    dynamodb = session.resource('dynamodb')

    table = dynamodb.Table('telegram-chat-history')

    USE_LLAMA = True

    if USE_LLAMA:
        def _predict(first_name, history_items, message):
            return predict(None, None, None, first_name, history_items, message, llama_generate)
    else:
        model, tokenizer, device = get_model()
        def _predict(first_name, history_items, message):
            return predict(model, tokenizer, device, first_name, history_items, message)
        
    print("Setting up the bot")

    send_commands(telegram_token)

    print("Started polling for messages...")

    while True:
        response = sqs_client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20
        )

        for message in response.get('Messages', []):
            # Ack messages sequentially
            try:
                # Ack the message first
                sqs_client.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )
                print("Handling message...", end="")
                start_time = time.time()
                handle_message(telegram_token, table, _predict, message)
                print("({:.4f}s)".format(time.time() - start_time))
            except Exception as e:
                logging.error(e)
                logging.error(message)

        time.sleep(1)