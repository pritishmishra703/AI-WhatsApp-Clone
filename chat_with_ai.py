import os
import re
import pandas as pd
import fireworks.client

######################################################################
# PLEASE SET BELOW VALUES
# SET FIREWORKS TOKEN
# GET IT HERE: https://fireworks.ai/api-keys
fireworks.client.api_key = '...'

# PLEASE CHANGE THIS WITH YOUR DIRECTORY WHICH CONTAINS ALL THE EXPORTED .TXT FILES
DATA_DIR = '...'

# THIS PERSON IS SENDING MESSAGES TO AI
SENDER_NAME = '...'

# THE AI WILL REPLY LIKE THIS PERSON
RESPOND_AS = '...'

# SETTING CHAT_NAME
# FOR PERSONAL DMs
CHAT_NAME = SENDER_NAME
# IF YOU WANT AI TO FEEL LIKE IT'S CHATTING IN SOME SPECIFIC WHATSAPP GROUP, PLEASE CHANGE CHAT_NAME WITH THAT GROUP'S NAME
# CHAT_NAME = 'group name goes here'

# FIREWORKS MODEL URL
# GET ACCOUNT ID HERE: https://fireworks.ai/users
account_id = '...'
# YOU CAN GET MODEL ID WHEN YOU RUN 'firectl list models'
model_id = '...'

FIREWORKS_MODEL_URL = f'accounts/{account_id}/models/{model_id}'
######################################################################

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_messages(chat_data):
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s*[APM]*M?) - ([^:]+): (.*?)(?=\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s*[APM]*M? - |$)'
    messages = re.findall(pattern, chat_data, re.DOTALL)
    messages = [[date, time, sender, message.strip()] for date, time, sender, message in messages]
    df = pd.DataFrame(messages, columns=['date', 'time', 'sender', 'message'])
    return df

if os.path.exists(DATA_DIR):
    chat_names = []
    senders = []
    for fname in os.listdir(DATA_DIR):
        chat_names.append(fname.replace('WhatsApp Chat with', '').replace('.txt', '').strip())
        msgs = load_text_file(os.path.join(DATA_DIR, fname))
        msgs_df = extract_messages(msgs)
        for sender in msgs_df['sender'].unique():
            if sender not in senders:
                senders.append(sender)
    
    if CHAT_NAME not in chat_names:
        raise ValueError(f"CHAT_NAME='{CHAT_NAME}' is incorrect. It should be one of {chat_names}")
    if SENDER_NAME not in senders:
        raise ValueError(f"SENDER_NAME='{SENDER_NAME}' is incorrect. It should be one of {senders}")
    if RESPOND_AS not in senders:
        raise ValueError(f"RESPOND_AS='{RESPOND_AS}' is incorrect. It should be one of {senders}")
else:
    raise ValueError("DATA_DIR doesn't exist. Please give correct path of DATA_DIR.")

# INITIALIZING MESSAGES LIST
messages = []

def get_reply(messages):
    prompt = f"<chat> {CHAT_NAME} </chat>"
    for msg in messages:
        prompt += f"\n<{msg['sender']}> {msg['content']} </{msg['sender']}>"
    prompt += f"\n<{RESPOND_AS}>"

    response = fireworks.client.Completion.create(
        model=FIREWORKS_MODEL_URL,
        prompt=prompt,
        max_tokens=500,
        stop=f"</{RESPOND_AS}>",
        temperature=0.3,
        top_k=10,
        repetition_penalty=1.1,
    )
    output_reply = response.choices[0].text.strip()
    return output_reply

while True:
    input_content = input(f'\n{SENDER_NAME}: ')
    messages.append({'sender': SENDER_NAME, 'content': input_content})
    output_msg = get_reply(messages)
    messages.append({'sender': RESPOND_AS, 'content': output_msg})
    output_msg = output_msg.replace('<br>', '\n\n')
    print(f'\n{RESPOND_AS}: {output_msg}')
