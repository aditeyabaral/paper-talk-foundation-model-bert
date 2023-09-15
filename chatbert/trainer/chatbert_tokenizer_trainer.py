import json
from pathlib import Path

from tqdm.auto import tqdm
from transformers import AutoTokenizer


class ChatBERTTokenizerTrainer:
    def __init__(self, model_path="bert-base-cased"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.add_tokens(
            [
                "[SENDER_TOKEN]",
                "[RECIPIENT_TOKEN]",
                "[MESSAGE_TOKEN]",

            ],
            special_tokens=True,
        )

    def train_tokenizer(self, messages_path, vocab_size=52000):
        messages_path = Path(messages_path).absolute()
        with open(messages_path, encoding="utf-8") as f:
            messages = json.load(f)
        user_ids = set()
        texts = list()
        for message in tqdm(messages):
            sender_id = message.get("sender_id")
            recipient_id = message.get("recipient_id")
            sender = message.get("sender")
            recipient = message.get("recipient")
            text = message.get("message")
            if sender_id is not None:
                user_ids.add(sender_id)
            if recipient_id is not None:
                user_ids.add(recipient_id)
            if sender is not None:
                texts.append(sender)
            if recipient is not None:
                texts.append(recipient)
            if text is not None:
                texts.append(text)

        user_ids = list(user_ids)
        user_ids = list(filter(lambda x: x is not None, user_ids))
        texts = list(filter(lambda x: x is not None, texts))
        self.tokenizer.add_tokens(user_ids, special_tokens=True)
        self.tokenizer.train_new_from_iterator(texts, vocab_size=vocab_size)

    def save_tokenizer(self, save_path):
        self.tokenizer.save_pretrained(save_path)

    def push_to_hub(self, tokenizer_path, commit_message, auth_token):
        self.tokenizer.push_to_hub(tokenizer_path, commit_message, use_auth_token=auth_token)
