# AI WhatsApp Clone

YouTube Video linked with this repository: https://youtu.be/hZ95S76uZvY

## Usage

### Downloading WhatsApp Chat History
To use this application, you need to download your WhatsApp chat history first. Follow these steps:

1. Open the WhatsApp app on your mobile device.
2. Go to the chat you want to download the history for.
3. Tap on the three-dot menu icon at the top-right corner of the chat.
4. Select "More" from the menu options.
5. Tap on "Export Chat".
6. Choose "Without Media" to exclude media files from the export (as the application currently cannot handle media).
7. Save the exported chat file to a location of your choice.
8. Repeat steps 2-7 for all the chats you want to download the history for.

Note: Providing more data will allow the AI to better understand and replicate your messaging style. The AI also recognizes the person it's conversing with and adjusts its personality accordingly, but it requires sufficient data to achieve this effectively.

After downloading the chat history from WhatsApp on your mobile device, transfer the compressed ZIP files to your laptop or PC. The subsequent steps should be performed on your laptop or PC:

1. Extract the ZIP files, and you will find a single .txt file for each chat.
2. Create a directory to store all the .txt files (e.g., `data/whatsapp_chat_history/`).
3. Keep all the .txt files in this directory.

For example, the directory structure could be like this:

```
data/whatsapp_chat_history/
   WhatsApp Chat with John Doe.txt
   WhatsApp Chat with Sarah Conner.txt
```

Please do not change the names of the .txt files; keep them as they were exported from WhatsApp.

### Formatting Chat Messages

To format the WhatsApp chat data for the language model, follow these steps:

1. Clone the repository: `git clone https://github.com/pritishmishra703/AI-WhatsApp-Clone.git`
2. Navigate to the cloned repository: `cd AI-WhatsApp-Clone`
3. Run the following command:

```
python chat_formatter.py --data_dir /path/to/data/dir --output_dir /path/to/output/dir --hf_token YOUR_HUGGINGFACE_TOKEN
```

Replace `/path/to/data/dir` with the directory where you stored the .txt files from the WhatsApp chat export (e.g., `data/whatsapp_chat_history/`).

Create a new directory for the output and provide its path for `--output_dir`. This directory should preferably be empty.

Replace `YOUR_HUGGINGFACE_TOKEN` with your actual Hugging Face token, which you can obtain from https://huggingface.co/settings/tokens. This token is required for accessing some gated models on Hugging Face.

Additional parameters:

- `--model_name`: The language model used for creating the AI WhatsApp Clone. The default is `meta-llama/Meta-Llama-3-8B-Instruct`. You can change it to another model of your choice, but note that this project has only been tested with the Llama-3-8b-Instruct model.
- `--max_context_length`: This parameter ensures that all the chats are within the specified token length. The default value is 2048, but you can increase it if needed.

After running the command, you will find two files in the output directory:

1. `whatsapp_chats_formatted.csv`
2. `whatsapp_chats_formatted.jsonl` (This is the important file we will use for training the language model using fireworks.ai)

Each line in the `whatsapp_chats_formatted.jsonl` file contains WhatsApp chats formatted in HTML format and limited to the specified `max_context_length`. Here's an example of how the formatted messages look:

```
<John> Hey! </John>
<Sarah> Hello <br>
What's up? </Sarah>
<John> Nothing </John>
```

The messages are formatted in HTML to create prompts for the language model. After training, you can provide the language model with text like:

```
<John> Hey! </John>
<Sarah>
```

The model will generate a reply, and you can control the generation by stopping it at the closing tag `</Sarah>`.

The `<br>` tag is used to separate multiple messages sent consecutively on WhatsApp. For example:

```
<John> Hi <br>
How are you? </John>
<Sarah> I'm good, thanks! <br>
And you? </Sarah>
```

This formatting allows the language model to understand the conversation flow and generate appropriate responses.

### Fine-Tuning Using Fireworks.ai


Download the fireworks command-line tool (firectl):

**Linux:**
```
wget -O firectl.gz https://storage.googleapis.com/fireworks-public/firectl/stable/linux-amd64.gz
gunzip firectl.gz
sudo install -o root -g root -m 0755 firectl /usr/local/bin/firectl
```

**macOS (Apple Silicon):**
```
curl https://storage.googleapis.com/fireworks-public/firectl/stable/darwin-arm64.gz -o firectl.gz
gzip -d firectl.gz && chmod a+x firectl
sudo mv firectl /usr/local/bin/firectl
sudo chown root: /usr/local/bin/firectl
```

**macOS (x86_64):**
```
curl https://storage.googleapis.com/fireworks-public/firectl/stable/darwin-amd64.gz -o firectl.gz
gzip -d firectl.gz && chmod a+x firectl
sudo mv firectl /usr/local/bin/firectl
sudo chown root: /usr/local/bin/firectl
```

**Windows:**

Try downloading from https://storage.googleapis.com/fireworks-public/firectl/stable/firectl.exe. If it doesn't work, it's recommended to use a cheap Ubuntu machine on AWS or another cloud provider and follow the Linux installation instructions.

2. First, create an account on [fireworks.ai](https://fireworks.ai) and then sign in to firectl:
```
firectl signin
```
A link will open, and you will be successfully signed in to firectl. Verify your sign-in by running:
```
firectl list accounts
```

3. Add funds to your fireworks.ai account from https://fireworks.ai/billing (minimum is $5). Pricing details can be found at https://fireworks.ai/pricing ($0.5 per 1 million tokens for models up to 16B parameters). _It costed me $2 for the fine-tuning_.

4. Upload the data:
```
firectl create dataset <DATASET_NAME> /path/to/whatsapp_chats_formatted.jsonl
```
Replace `<DATASET_NAME>` with your desired dataset name, and provide the correct path to the `whatsapp_chats_formatted.jsonl` file in your output directory.

5. Create a `settings.yaml` file in your output directory with the following content:
```
dataset: <DATASET_NAME>
epochs: 1.0
lora_rank: 32
model_id: aiwhatsappclone

text_completion:
  input_template: ""
  output_template: "{text}"

base_model: meta-llama/Meta-Llama-3-8B-Instruct
```
Replace `<DATASET_NAME>` with the name you chose in the previous step. You can modify the hyperparameters according to your requirements. Find a list of all available hyperparameters at https://readme.fireworks.ai/docs/fine-tuning-models#additional-tuning-options.

6. Kick off the fine-tuning job:
```
firectl create fine-tuning-job --settings-file path/to/settings.yaml --display-name "AI WhatsApp Clone"
```

7. Copy the "JOB ID" from the output of the previous command. Check the status of the fine-tuning job using:
```
firectl get fine-tuning-job <JOB_ID>
```
The "State" field in the output will indicate the current status (CREATING, RUNNING, etc.). Wait until the fine-tuning is complete.

8. After the fine-tuning is complete, list your models:
```
firectl list models
```
If your model is listed, it has been successfully trained.

9. Deploy the model:
```
firectl deploy <MODEL_ID>
```
Replace `<MODEL_ID>` with the ID of your model from the list models command.

10. Wait a few seconds and run `firectl list models` again. If the status of your model is "DEPLOYED," it is ready to use.

### Chat With Your AI Clone

After successfully fine-tuning and deploying your model, you can interact with your AI clone by following these steps:

1. Navigate to the directory where you cloned the repository: `cd /path/to/cloned/repository`

2. Open the `chat_with_ai.py` file in a text editor.

3. Follow the comments in the code to fill in the required values, such as your Fireworks API token, the deployed model ID, etc.

4. Save the changes to the `chat_with_ai.py` file.

5. Run the file:
   ```
   python chat_with_ai.py
   ```

6. You can now start typing your messages and hit Enter. The AI clone will generate responses based on the fine-tuned model.

7. Experiment with different temperature and top_k values to find the combination that works best for you. According to my experimentations, a temperature between 0.2 to 0.5 and a top_k between 10 to 50 may produce desirable results.

If you encounter any errors during this process, feel free to open an issue in the repository, and we will do our best to assist you in resolving the issue.

By following these steps, you can engage in conversations with your AI clone, which has been fine-tuned on your WhatsApp chat history to mimic your communication style and personality.
