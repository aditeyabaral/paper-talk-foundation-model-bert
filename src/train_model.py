import torch

from chatbert import ChatBERT, ChatBERTDataset
from chatbert.trainer import ChatBERTTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChatBERT(
    config_path="bert-base-cased",
    tokenizer_path="aditeyabaral/paper-talk-tokenizer", # Replace with your username
    context_selector_mode="keybert",
    device="cuda"
).to(device)

dataset = ChatBERTDataset(
    conversational_data_filepath="data/messages.json",
    membership_data_filepath="data/groups.json",
    member_ids_filepath="data/members.json",
)
dataset.process_conversational_data(model.mlm_tokenizer)
all_member_ids = dataset.member_ids_data["ids"]
model.add_special_tokens_to_tokenizer(all_member_ids)

trainer = ChatBERTTrainer(
    model=model,
    learning_rate=1e-6,
    optimizer_class="adamw",
    scheduler_class="plateau",
)

trainer.train(
    dataset=dataset,
    model=model,
    batch_size=2,
    learning_rate=1e-6,
    epochs=10,
    alpha=1.0,
    beta=0.5,
    gamma=0.5,
    delta=0.5,
    eta=0.5,
    zeta=0.5,
    freeze_components=[],
    train_components=[
        "mlm",
        "topic",
        "membership",
        "sender_cls",
        "recipient_cls",
        "next_sentence_prediction"
    ],
    save_dir="./saved_models/",
    save_every=1,
    save_latest=True,
    save_model_name="chatbert_model.pt",
    upload_model_to_hub=False,
    use_tensorboard=True,
    tensorboard_log_dir="./logs",
    auth_token=None,
    hub_model_name="chatbert-bert-base-cased",
    hub_organization="chatbert",
    device=device,
)
