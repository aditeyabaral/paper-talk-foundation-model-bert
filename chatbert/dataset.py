import json
import logging

from tqdm.auto import tqdm


class ChatBERTDataset:
    def __init__(self, conversational_data_filepath, membership_data_filepath, member_ids_filepath, **kwargs):
        self.conversational_training_examples = None
        self.conversational_data = json.load(open(conversational_data_filepath))
        self.membership_data = json.load(open(membership_data_filepath))
        self.member_ids_data = json.load(open(member_ids_filepath))

        if "member_limit" in kwargs:
            self.membership_data = {
                key: data for key, data in self.membership_data.items() if
                len(self.membership_data[key]["member_ids"]) <= kwargs["member_limit"]
            }

    def process_conversational_data(self, tokenizer, **kwargs):
        self.add_special_tokens_to_conversational_data()
        self.get_tokenized_training_sequences(tokenizer, **kwargs)

    def add_special_tokens_to_conversational_data(self):
        self.conversational_data = list(
            filter(lambda d: d["message"].strip(), self.conversational_data)
        )
        self.conversational_data = list(
            map(
                lambda d: {
                    **d,
                    "text": f"{d['sender_id']} [RECIPIENT_TOKEN] {d['recipient_id']} [MESSAGE_TOKEN] {d['message']}",
                },
                self.conversational_data,
            )
        )
        logging.debug(self.conversational_data[:3])
        logging.info(f"Added special tokens to {len(self.conversational_data)} conversational data")

    def get_tokenized_training_sequences(self, tokenizer, **kwargs):
        texts = [message["text"] for message in self.conversational_data]
        conversational_data_encoded = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=kwargs.get("max_length", 256),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_token_type_ids=True,
        )
        logging.debug(conversational_data_encoded.items())
        logging.info(f"Tokenized {len(conversational_data_encoded)} conversational data")
        self.get_conversational_training_examples(conversational_data_encoded)

    def get_conversational_training_examples(self, conversational_data_encoded):
        self.conversational_training_examples = list()
        for i in tqdm(range(conversational_data_encoded["input_ids"].shape[0])):
            input_ids = conversational_data_encoded["input_ids"][i]
            attention_mask = conversational_data_encoded["attention_mask"][i]
            special_tokens_mask = conversational_data_encoded["special_tokens_mask"][i]
            token_type_ids = conversational_data_encoded["token_type_ids"][i]
            self.conversational_training_examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "special_tokens_mask": special_tokens_mask,
                    "token_type_ids": token_type_ids,
                }
            )
        logging.info(f"Obtained {len(self.conversational_training_examples)} conversational training examples")
