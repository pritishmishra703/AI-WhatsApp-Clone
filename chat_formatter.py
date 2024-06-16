import os
import re
import json
import argparse
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

class WhatsAppChatParser:
    def __init__(self, model_name='meta-llama/Meta-Llama-3-8B', hf_token=None):
        """
        Initializes the WhatsAppChatParser for parsing and formatting WhatsApp chat data.

        Args:
            model_name (str): The name of the model to use for tokenization.
            hf_token (str): The Hugging Face token for authentication.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    def load_text_file(self, file_path):
        """
        Loads text content from a file.

        Args:
            file_path (str): The path to the text file.

        Returns:
            str: The content of the text file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def filter_messages(self, df):
        """
        Filters out media and deleted messages from the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing chat messages.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        df = df[~df['message'].str.contains('<Media omitted>')]
        df = df[~df['message'].str.contains('deleted this message', case=False)]
        df = df[~df['message'].str.contains('message was deleted', case=False)]
        return df.reset_index(drop=True)

    def extract_messages(self, chat_data):
        """
        Extracts messages from the chat data using regex pattern.

        Args:
            chat_data (str): The raw chat data.

        Returns:
            pd.DataFrame: A DataFrame containing extracted messages.
        """
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s*[APM]*M?) - ([^:]+): (.*?)(?=\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s*[APM]*M? - |$)'
        messages = re.findall(pattern, chat_data, re.DOTALL)
        messages = [[date, time, sender, message.strip()] for date, time, sender, message in messages]
        df = pd.DataFrame(messages, columns=['date', 'time', 'sender', 'message'])
        return self.filter_messages(df)

    def calculate_token_length(self, text):
        """
        Calculates the number of tokens in the text using the tokenizer.

        Args:
            text (str): The input text.

        Returns:
            int: The number of tokens.
        """
        return len(self.tokenizer(text)['input_ids'])

    def format_chat(self, df, chat_name, max_context_length):
        """
        Formats the DataFrame for model input with context length constraints.

        Args:
            df (pd.DataFrame): The DataFrame containing chat messages.
            chat_name (str): The name of the chat.
            max_context_length (int): The maximum context length for the model.

        Returns:
            pd.DataFrame: A DataFrame containing formatted chat data.
        """
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y')
        grouped = df.groupby(df['date'].dt.date)

        formatted_data = []
        for date, group in tqdm(grouped, total=len(grouped)):
            day_chunks = []
            current_chunk = f"<chat> {chat_name} </chat>\n"
            previous_sender = None
            chunk_start_time = None

            for _, row in group.iterrows():
                sender, message = row['sender'], row['message']
                if chunk_start_time is None:
                    chunk_start_time = row['time']
                if sender == previous_sender:
                    message_block = f" <br>\n{message}"
                else:
                    if previous_sender is not None:
                        message_block = f" </{previous_sender}>\n<{sender}> {message}"
                    else:
                        message_block = f"<{sender}> {message}"
                    previous_sender = sender

                if self.calculate_token_length(current_chunk + message_block) < max_context_length:
                    current_chunk += message_block
                else:
                    day_chunks.append((current_chunk + f" </{previous_sender}>", chunk_start_time))
                    current_chunk = f"<chat> {chat_name} </chat>\n<{sender}> {message}"
                    previous_sender = sender
                    chunk_start_time = row['time']
            
            if current_chunk:
                day_chunks.append((current_chunk + f" </{previous_sender}>", chunk_start_time))
            
            for chunk, start_time in day_chunks:
                formatted_data.append({'date': date, 'time': start_time, 'chat_name': chat_name, 'text': chunk})

        return pd.DataFrame(formatted_data)

def main(data_dir, output_path, model_name, hf_token, max_context_length):
    """
    Main function to parse and format WhatsApp chat data for model input.

    Args:
        data_dir (str): The directory containing WhatsApp chat history files.
        output_path (str): The path to the output CSV file.
        model_name (str): The name of the model to use for tokenization.
        hf_token (str): The Hugging Face token for authentication.
    """
    parser = WhatsAppChatParser(model_name=model_name, hf_token=hf_token)
    
    chat_files = [file for file in os.listdir(data_dir) if file.endswith('.txt')]
    parsed_chat_dfs = []
    
    for file_name in chat_files:
        file_path = os.path.join(data_dir, file_name)
        chat_data = parser.load_text_file(file_path)
        parsed_chat_dfs.append(parser.extract_messages(chat_data))

    formatted_dfs = []
    for file_name, df in tqdm(zip(chat_files, parsed_chat_dfs), total=len(parsed_chat_dfs)):
        chat_name = file_name.replace('.txt', '').replace('WhatsApp Chat with', '').strip()
        formatted_df = parser.format_chat(df, chat_name, max_context_length)
        formatted_dfs.append(formatted_df)

    output_csv_path = os.path.join(output_path, 'whatsapp_chats_formatted.csv')
    output_jsonl_path = os.path.join(output_path, 'whatsapp_chats_formatted.jsonl')

    combined_df = pd.concat(formatted_dfs).sort_values(by=['date', 'time']).reset_index(drop=True)
    combined_df.to_csv(output_csv_path, index=False)

    counter = 0
    with open(output_jsonl_path, "w") as f:
        for _, row in combined_df.iterrows():
            json.dump({'text': row.text}, f)
            counter += f.write("\n")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Format WhatsApp chat data for LLM")
    arg_parser.add_argument(
        '--data_dir', 
        type=str, 
        default='data/whatsapp_chat_history', 
        help="Directory containing WhatsApp chat history (.txt) files"
    )
    arg_parser.add_argument(
        '--output_dir', 
        type=str, 
        default='data/output', 
        help="Path to the output directory"
    )
    arg_parser.add_argument(
        '--model_name', 
        type=str, 
        default='meta-llama/Meta-Llama-3-8B-Instruct', 
        help="Model name for tokenization"
    )
    arg_parser.add_argument(
        '--hf_token', 
        type=str, 
        required=True,
        help="Hugging Face token for authentication"
    )
    arg_parser.add_argument(
        '--max_context_length', 
        type=int, 
        default=2048,
        help="Max context length for sequences"
    )
    args = arg_parser.parse_args()

    def check_directory_exists(directory):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist. Please create it or provide another directory.")
        
    check_directory_exists(args.data_dir)
    check_directory_exists(args.output_dir)

    main(
        data_dir=args.data_dir, 
        output_path=args.output_dir, 
        model_name=args.model_name, 
        hf_token=args.hf_token, 
        max_context_length=args.max_context_length
    )
