import random
import re

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TopicAssociation(nn.Module):
    def __init__(
            self,
            selector_mode="keybert",
            device="cpu",
            **kwargs
    ):
        super(TopicAssociation, self).__init__()
        self.device = device
        self.selector_mode = selector_mode

        if self.selector_mode == "keybert":
            self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)
            self.keybert = KeyBERT(model=self.sentence_transformer)
        elif self.selector_mode == "context":
            if "selector_model" not in kwargs or "selector_tokenizer" not in kwargs:
                raise ValueError("selector_model and selector_tokenizer must be provided for context selector")
            self.selector_model = kwargs["selector_model"]
            self.selector_tokenizer = kwargs["selector_tokenizer"]
            self.c = nn.Parameter(torch.rand(self.selector_model.config.vocab_size)).to(self.device)
            self.W = nn.Linear(self.selector_model.config.hidden_size, 1).to(self.device)
            self.sigmoid = nn.Sigmoid().to(self.device)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    @staticmethod
    def filter_text(text):
        text = text.lower()
        stopwords = list(nltk.corpus.stopwords.words("english"))
        stopwords_regex_string = r"\b(" + "|".join(stopwords) + ")\\b"
        text = re.sub(stopwords_regex_string, "", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text

    def forward_whole_word_selection(self, messages, preprocess=False, threshold=0.5):
        if preprocess:
            messages = list(map(self.filter_text, messages))

        embedding_matrix = self.selector_model.get_input_embeddings()._parameters['weight'].to(self.device)
        context_keywords = list()
        loss_cs = list()
        for message in messages:
            token_ids_map = {
                word: self.selector_tokenizer.encode(
                    word,
                    add_special_tokens=False
                ) for word in message.split()
            }
            input_ids = list(token_ids_map.values())
            input_ids = np.asarray([item for sublist in input_ids for item in sublist])

            word_embedding_matrix = embedding_matrix[input_ids, :].to(self.device)
            embedding_dim = word_embedding_matrix.shape[1]
            embedding_dim_root = embedding_dim ** 0.5
            self_attention_matrix = (
                    torch.matmul(word_embedding_matrix, word_embedding_matrix.T) / embedding_dim_root
            )
            self_attention_matrix = F.softmax(self_attention_matrix, dim=0)
            embedding_attention_matrix = torch.matmul(
                self_attention_matrix, word_embedding_matrix
            ).to(self.device)

            q = self.sigmoid(self.W(embedding_attention_matrix).flatten())
            c_subset = self.c[torch.LongTensor(input_ids)]
            input_ids_selection = (c_subset >= threshold).tolist()

            selected_input_ids = input_ids[input_ids_selection]
            current_context_keywords = list()
            for selected_input_id in selected_input_ids:
                for token in token_ids_map:
                    if selected_input_id in token_ids_map[token] and token not in current_context_keywords:
                        current_context_keywords.append(token)
                        break
            # current_context_keywords = list(map(str.lower, current_context_keywords))
            random.shuffle(current_context_keywords)
            context_keywords.append(current_context_keywords)
            loss = F.kl_div(q.log(), c_subset)
            loss_cs.append(loss)
        loss_cs = torch.stack(loss_cs).mean()
        return loss_cs, context_keywords

    def forward_token_selection(self, messages, preprocess=False, threshold=0.5):
        if preprocess:
            messages = list(map(self.filter_text, messages))

        embedding_matrix = self.selector_model.get_input_embeddings()._parameters['weight'].to(self.device)
        context_keywords = list()
        loss_cs = list()
        for message in messages:
            input_ids = self.selector_tokenizer.encode(message, add_special_tokens=False)

            word_embedding_matrix = embedding_matrix[input_ids, :].to(self.device)
            embedding_dim = word_embedding_matrix.shape[1]
            embedding_dim_root = embedding_dim ** 0.5
            self_attention_matrix = (
                    torch.matmul(word_embedding_matrix, word_embedding_matrix.T) / embedding_dim_root
            )
            self_attention_matrix = F.softmax(self_attention_matrix, dim=0)
            embedding_attention_matrix = torch.matmul(
                self_attention_matrix, word_embedding_matrix
            ).to(self.device)

            q = self.sigmoid(self.W(embedding_attention_matrix).flatten())
            c_subset = self.c[torch.LongTensor(input_ids)]
            input_ids_selection = (c_subset >= threshold).tolist()

            selected_input_ids = np.asarray(input_ids)[input_ids_selection]
            current_context_keywords = list()
            for selected_input_id in selected_input_ids:
                current_context_keywords.append(self.selector_tokenizer.decode(selected_input_id))
            # current_context_keywords = list(map(str.lower, current_context_keywords))
            random.shuffle(current_context_keywords)
            context_keywords.append(current_context_keywords)
            loss = F.kl_div(q.log(), c_subset)
            loss_cs.append(loss)

        loss_cs = torch.stack(loss_cs).mean()
        return loss_cs, context_keywords

    def forward_keybert_selection(self, messages, preprocess=False):
        if preprocess:
            messages = list(map(self.filter_text, messages))

        all_keywords = self.keybert.extract_keywords(
            messages,
            keyphrase_ngram_range=(1, 3),
            use_maxsum=True,
            top_n=5
        )

        if not all_keywords:
            return list()

        keywords = list()
        if isinstance(all_keywords[0], tuple):
            keywords = [keyword[0] for keyword in all_keywords]
        else:
            for all_keyword_list in all_keywords:
                all_keyword_list = [keyword[0] for keyword in all_keyword_list]
                keywords.append(all_keyword_list)
        return keywords

    def forward_sentence_selection(self, messages, preprocess=True):
        if preprocess:
            messages = list(map(self.filter_text, messages))
        return messages

    def forward(
            self,
            messages,
            preprocess=True,
            threshold=0.5
    ):
        if self.selector_mode == "whole-word":
            return self.forward_whole_word_selection(
                messages, preprocess, threshold
            )
        elif self.selector_mode == "token":
            return self.forward_token_selection(
                messages, preprocess, threshold
            )
        elif self.selector_mode == "keybert":
            return self.forward_keybert_selection(messages, preprocess)
        elif self.selector_mode == "sentence":
            return self.forward_sentence_selection(messages, preprocess)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
