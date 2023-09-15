from pathlib import Path

from chatbert.trainer import ChatBERTTokenizerTrainer

tokenizer_trainer = ChatBERTTokenizerTrainer(model_path="bert-base-cased")
tokenizer_trainer.train_tokenizer(
    messages_path=Path("data/messages.json"),
    vocab_size=52000,
)
tokenizer_trainer.save_tokenizer("tokenizer")

tokenizer_trainer.push_to_hub(
    tokenizer_path="<your_username>/tokenizer",
    commit_message="Added tokenizer trained on WhatsApp messages",
    auth_token="<your_auth_token>"
)
