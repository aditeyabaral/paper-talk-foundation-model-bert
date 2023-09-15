import random
from pathlib import Path
from typing import List, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import sent_tokenize
from transformers import BertConfig, BertTokenizer, BertForMaskedLM

from .topic_association import TopicAssociation


class ChatBERTOutput:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class ChatBERT(nn.Module):
    def __init__(
            self,
            config_path="bert-base-cased",
            tokenizer_path="chatbert/tokenizer",
            context_selector_mode="keybert",
            device="cuda"
    ):
        super(ChatBERT, self).__init__()
        self.device = device
        self.config_path = config_path
        self.tokenizer_path = tokenizer_path
        self.context_selector_mode = context_selector_mode

        if self.config_path is None:
            self.bert_config = BertConfig()
        else:
            self.bert_config = BertConfig.from_pretrained(self.config_path)
        self.mlm = BertForMaskedLM(self.bert_config).to(self.device)
        self.mlm_tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
        self.mlm.resize_token_embeddings(len(self.mlm_tokenizer))

        self.topic_selector = TopicAssociation(
            selector_model=self.mlm.base_model,
            selector_tokenizer=self.mlm_tokenizer,
            selector_mode=context_selector_mode,
            device=self.device
        ).to(self.device)

        self.sender_classifier = nn.Sequential(
            nn.Linear(self.mlm.base_model.config.hidden_size * 3, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.recipient_classifier = nn.Sequential(
            nn.Linear(self.mlm.base_model.config.hidden_size * 3, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.next_sentence_classifier = nn.Sequential(
            nn.Linear(self.mlm.base_model.config.hidden_size * 3, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(self.device)

    def freeze(self):
        for param in self.mlm.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.mlm.parameters():
            param.requires_grad = True

    def push_to_hub(self, repo_name: str, commit_message: str, auth_token: str):
        self.mlm.base_model.push_to_hub(
            repo_name,
            commit_message=f"Model: {commit_message}",
            use_auth_token=auth_token
        )
        self.mlm_tokenizer.push_to_hub(
            repo_name,
            commit_message=f"Tokenizer: {commit_message}",
            use_auth_token=auth_token
        )

    def add_special_tokens_to_tokenizer(self, special_tokens: List[str]):
        self.mlm_tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.mlm.resize_token_embeddings(len(self.mlm_tokenizer))

    def encode_tokens_for_embedding(self, tokens: List[str], max_length: int = 32):
        encoding = self.mlm_tokenizer.batch_encode_plus(
            tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        return encoding

    def minimise_distance_between_words(self, words: List[str]):
        encoding = self.encode_tokens_for_embedding(words)
        embeddings = self.mlm.base_model(**encoding).last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=1, dim=1)
        embeddings = 1 - torch.matmul(embeddings, embeddings.T)
        embeddings = embeddings[torch.triu(torch.ones(embeddings.size()), diagonal=1).bool()]
        return embeddings.mean()

    def minimise_distance_between_words_encodings(self, words):
        embeddings = self.mlm.base_model(**words).last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=1, dim=1)
        embeddings = 1 - torch.matmul(embeddings, embeddings.T)
        embeddings = embeddings[torch.triu(torch.ones(embeddings.size()), diagonal=1).bool()]
        return embeddings.mean()

    def minimise_distance_between_word_and_context_lists(self, words: List[str], context: List[str]):
        encoding = self.encode_tokens_for_embedding(context)
        context_embeddings = self.mlm.base_model(**encoding).last_hidden_state.mean(dim=1)
        context_embeddings = F.normalize(context_embeddings, p=1, dim=1)

        encoding = self.encode_tokens_for_embedding(words)
        words_embeddings = self.mlm.base_model(**encoding).last_hidden_state.mean(dim=1)
        words_embeddings = F.normalize(words_embeddings, p=1, dim=1)

        embeddings = 1 - torch.matmul(words_embeddings, context_embeddings.T)
        return embeddings.mean()

    def minimise_distance_between_word_and_context_encodings(self, words, context):
        context_embeddings = self.mlm.base_model(**context).last_hidden_state.mean(dim=1)
        context_embeddings = F.normalize(context_embeddings, p=1, dim=1)
        words_embeddings = self.mlm.base_model(**words).last_hidden_state.mean(dim=1)
        words_embeddings = F.normalize(words_embeddings, p=1, dim=1)
        embeddings = 1 - torch.matmul(words_embeddings, context_embeddings.T)
        return embeddings.mean()

    def forward_classifier_with_2_inputs(
            self,
            input_tokens_1,
            input_tokens_2,
            labels,
            classifier,
            max_lengths=(256, 128)
    ):
        labels = labels.type(torch.float32)
        encoding = self.mlm_tokenizer.batch_encode_plus(
            input_tokens_1,
            padding="max_length",
            max_length=max_lengths[0],
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        embedding1 = self.mlm.base_model(**encoding).last_hidden_state.mean(dim=1)
        encoding = self.mlm_tokenizer.batch_encode_plus(
            input_tokens_2,
            padding="max_length",
            max_length=max_lengths[1],
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.device)
        embedding2 = self.mlm.base_model(**encoding).last_hidden_state.mean(dim=1)
        difference = torch.abs(embedding1 - embedding2)
        embedding = torch.cat((embedding1, embedding2, difference), dim=1)
        logits = classifier(embedding).flatten()
        loss = F.binary_cross_entropy(logits, labels)
        return loss

    def forward_classifier_with_inputs(self, labels, classifier, *input_tokens):
        labels = labels.type(torch.float32)
        embeddings = list()
        for input_token in input_tokens:
            encoding = self.mlm_tokenizer.batch_encode_plus(
                input_token,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            ).to(self.device)
            embedding = self.mlm.base_model(**encoding).last_hidden_state.mean(dim=1)
            embeddings.append(embedding)
        embedding = torch.cat(embeddings, dim=1)
        logits = classifier(embedding).flatten()
        loss = F.binary_cross_entropy(logits, labels)
        return loss

    # TODO: Move to dataset.py
    def get_sender_receiver_classification_inputs_labels(self, batch, key, negative_sampling=True):
        if negative_sampling:
            inputs = list()
            all_ids = list(set([conversation[key] for conversation in batch]))
            len_all_ids = len(all_ids)
            for idx, conversation in enumerate(batch):
                random_id = random.choice(all_ids)
                if len_all_ids > 1:
                    while random_id == conversation[key]:
                        random_id = random.choice(all_ids)
                    random_label = 0
                else:
                    random_label = 1

                inputs.append(
                    [
                        conversation[key],
                        conversation["message"],
                        1
                    ]
                )
                inputs.append(
                    [
                        random_id,
                        conversation["message"],
                        random_label
                    ]
                )
            random.shuffle(inputs)
            sender_id_list, message_list, labels = zip(*inputs)
            labels = torch.tensor(labels).to(self.device)
        else:
            sender_id_list = list()
            message_list = list()
            labels = torch.tensor([1.0 for _ in range(len(batch))]).to(self.device)
            for conversation in batch:
                sender_id_list.append(conversation[key])
                message_list.append(conversation["message"])

        return sender_id_list, message_list, labels

    # TODO: Move to dataset.py
    def get_next_sentence_prediction_inputs_labels(self, batch, negative_sampling=True):
        sentence_lists = [sent_tokenize(conversation["message"]) for conversation in batch]
        first_sentences = list()
        second_sentences = list()
        labels = list()
        for sentence_list in sentence_lists:
            num_sentences = len(sentence_list)
            if num_sentences > 1:
                all_indices = list(range(num_sentences))
                for i in range(num_sentences - 1):
                    first_sentences.append(sentence_list[i])
                    second_sentences.append(sentence_list[i + 1])
                    labels.append(1)
                    if negative_sampling and num_sentences > 2:
                        random_idx = random.choice(all_indices[:i] + all_indices[i + 2:])
                        first_sentences.append(sentence_list[i])
                        second_sentences.append(sentence_list[random_idx])
                        labels.append(0)
        if first_sentences and second_sentences:
            labels = torch.tensor(labels).to(self.device)
        return first_sentences, second_sentences, labels

    def forward_mlm(self, mlm_batch: Dict[str, torch.Tensor]):
        outputs = self.mlm(**mlm_batch)
        return outputs.loss

    def forward_topic_association(self, topic_association_batch: List[Dict[str, str]]):
        topic_association_loss = 0
        loss_cs = None
        total_examples = 0
        messages = [conversation["message"] for conversation in topic_association_batch]

        result = self.topic_selector(messages)
        if self.context_selector_mode in ("keybert", "sentence"):
            context_keywords = result
        else:
            loss_cs, context_keywords = result

        if not context_keywords:
            return None, None

        for idx, conversation in enumerate(topic_association_batch):
            if context_keywords[idx]:
                if isinstance(context_keywords[idx], str):
                    context_keywords[idx] = [context_keywords[idx]]
                context_encoding = self.mlm_tokenizer.batch_encode_plus(
                    context_keywords[idx],
                    padding="max_length",
                    max_length=32,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(self.device)
                sender_recipient_encoding = self.mlm_tokenizer.batch_encode_plus(
                    [conversation["sender_id"], conversation["recipient_id"]],
                    padding="max_length",
                    max_length=32,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(self.device)
                topic_association_loss += self.minimise_distance_between_word_and_context_encodings(
                    sender_recipient_encoding, context_encoding
                )
                total_examples += 1
        if total_examples > 0:
            topic_association_loss = topic_association_loss / total_examples
        else:
            topic_association_loss = None
        return topic_association_loss, loss_cs

    def forward_membership(self, membership_batch: List[Dict[str, Any]], threshold_membership_size: int = 100):
        membership_loss = 0
        num_memberships = 0
        for membership in membership_batch:
            if len(membership["member_ids"]) >= threshold_membership_size:
                continue
            title = membership["title"]
            member_ids = membership["member_ids"]
            member_id_title_encoding = self.mlm_tokenizer.batch_encode_plus(
                [title] + member_ids,
                add_special_tokens=False,
                max_length=32,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            membership_loss += self.minimise_distance_between_words_encodings(member_id_title_encoding)
            num_memberships += 1
        return membership_loss / num_memberships

    def forward_sender_classification(
            self,
            sender_classification_batch: List[Dict[str, Any]],
            negative_sampling: bool = True
    ):
        sender_id_list, message_list, labels = self.get_sender_receiver_classification_inputs_labels(
            sender_classification_batch, "sender_id", negative_sampling
        )
        loss = self.forward_classifier_with_2_inputs(sender_id_list, message_list, labels, self.sender_classifier)
        return loss

    def forward_recipient_classification(
            self,
            recipient_classification_batch: List[Dict[str, Any]],
            negative_sampling: bool = True
    ):
        recipient_id_list, message_list, labels = self.get_sender_receiver_classification_inputs_labels(
            recipient_classification_batch, "recipient_id", negative_sampling
        )
        loss = self.forward_classifier_with_2_inputs(recipient_id_list, message_list, labels, self.recipient_classifier)
        return loss

    def forward_next_sentence_prediction(
            self,
            next_sentence_prediction_batch: List[Dict[str, Any]],
    ):
        first_sentences, second_sentences, labels = next_sentence_prediction_batch
        if first_sentences and second_sentences:
            total_pairs = len(first_sentences)
            loss = 0
            for i in range(0, total_pairs, 8):
                first_sentences_batch = first_sentences[i:i + 8]
                second_sentences_batch = second_sentences[i:i + 8]
                labels_batch = labels[i:i + 8]
                loss += self.forward_classifier_with_2_inputs(
                    first_sentences_batch,
                    second_sentences_batch,
                    labels_batch,
                    self.next_sentence_classifier,
                    max_lengths=(128, 128)
                )
            loss = loss / total_pairs
        else:
            loss = None
        return loss

    def forward(self, **kwargs):
        mlm_batch = kwargs.get("mlm_batch")
        membership_batch = kwargs.get("membership_batch")
        topic_association_batch = kwargs.get("topic_association_batch")
        sender_cls_batch = kwargs.get("sender_cls_batch")
        recipient_cls_batch = kwargs.get("recipient_cls_batch")
        next_sentence_prediction_batch = kwargs.get("next_sentence_prediction_batch")
        negative_sampling = kwargs.get("negative_sampling", True)
        output = ChatBERTOutput()

        if mlm_batch:
            output.loss_mlm = self.forward_mlm(mlm_batch)
        if membership_batch:
            output.loss_membership = self.forward_membership(membership_batch)
        if topic_association_batch:
            loss_topic_association, loss_cs = self.forward_topic_association(topic_association_batch)
            if loss_cs is not None:
                output.loss_cs = loss_cs
            if loss_topic_association is not None:
                output.loss_topic_association = loss_topic_association
        if sender_cls_batch:
            output.loss_sender_cls = self.forward_sender_classification(sender_cls_batch, negative_sampling)
        if recipient_cls_batch:
            output.loss_recipient_cls = self.forward_recipient_classification(recipient_cls_batch, negative_sampling)
        if next_sentence_prediction_batch:
            loss_nsp = self.forward_next_sentence_prediction(next_sentence_prediction_batch)
            if loss_nsp is not None:
                output.loss_nsp = loss_nsp

        return output

    def save(self, path: Union[str, Path]):
        if Path(path).parent.exists():
            torch.save(
                {
                    "model_state_dict": self.state_dict(),
                    "config_path": self.config_path,
                    "tokenizer_path": self.tokenizer_path,
                    "context_selector_mode": self.context_selector_mode,
                },
                path
            )
        else:
            raise FileNotFoundError

    def load(self, path: Union[str, Path]):
        if not Path(path).exists():
            raise FileNotFoundError
        checkpoint = torch.load(path, map_location=self.device)
        self.config_path = checkpoint["config_path"]
        self.tokenizer_path = checkpoint["tokenizer_path"]
        self.context_selector_mode = checkpoint["context_selector_mode"]
        self.load_state_dict(checkpoint["model_state_dict"])
