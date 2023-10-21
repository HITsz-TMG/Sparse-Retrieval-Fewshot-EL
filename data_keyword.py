import copy
import json
import os.path

import pandas as pd
import torch.nn.utils.rnn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from bm25 import BM25
import utils
from collections import Counter, defaultdict
from utils import remove_punctuation

REMOVED_ELECTRA_TOKENS = ['##ing', 'after', 'to', 'which', 'under', 'for', 'this', 'of', 's', 'one', 'and', 'by', 'up',
                          'him', 'he', 'an', 'with', 'a', 'other', 'the', 'its', 'was', 'on', 'his', 'from', 'not',
                          'in',
                          'used', 'then', 'that', 'all', 'who', 'however', 'been', '##s', 'had', 'them', 'be', 'it',
                          'world',
                          'first', 'at', 'new', 'they', 'time', 'out', 'during', 'war', 'as', 'also', 'is', 'later',
                          'are', 'two', 'but',
                          'there', 'while', 'into', 'when', 'would', 'were', 'where', 'or', 'have', 'three', 'their',
                          'before', 'only', 'has', 'part', 'over', 'battle', '##r', '1']

REMOVED_TOKENS = ['on', 'and', 'the', 'not', 'to', 's', 'up', 'under', 'from', 'him', 'he', 'with', 'his', 'this', 'of',
                  'for', 'one', 'after', 'by', 'other', 'an', 'which', 'was', 'its', 'a', 'it', 'had', 'who', 'that',
                  'then', 'been', 'them', 'all', 'used', 'in', 'however', 'during', 'as', 'also', 'time', 'first',
                  'war', 'new', 'at', 'they', 'world', 'out', 'is', 'where', 'their', 'were', 'but', 'later', 'only',
                  'while',
                  'three', 'before', 'be', 'would', 'when', 'there', 'into', 'have', 'two', 'are', 'or', 'battle',
                  'over', '1']


def read_data(path):
    with open(path, encoding="utf-8") as f:
        data = [json.loads(x) for x in f]
    return data


def read_kb(path):
    with open(path, encoding="utf-8") as f:
        data = json.loads(f.read())

    return data


def write_data(data, path):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in data)


def get_entity_index_map(kb):
    keys = list(kb.keys())
    entity_index_map = defaultdict(int)
    for index, key in enumerate(keys):
        entity_index_map[key] = index
    return entity_index_map


def process_train_data(samples_train, kb_train, tokenizer, max_text_len,
                       label_num):
    text_input_tokens, labels, train_selected_words = [], [], []
    corpus = []
    entity_index_map = get_entity_index_map(kb_train)

    for des in tqdm(kb_train.values()):
        clear_des = remove_punctuation(des["text"]).lower()
        tokens = clear_des.split(" ")
        clear_des_tokens = clear_tokens(tokens)
        corpus.append(clear_des_tokens)
    train_bm25 = BM25(corpus)
    for sample_train in samples_train:
        text = sample_train["text"].lower()
        kb_id = sample_train["kb_id"]
        kb_id_index = entity_index_map[kb_id]
        text_words = text.split(" ")
        start, end = text_words.index("[e1]"), text_words.index("[\e1]")
        half_text_len = (max_text_len - end + start) // 2
        text_words = text_words[max(0, start - half_text_len):end + half_text_len]
        text_words_scores = train_bm25.get_words_score(text_words, kb_id_index)

        seleted_words = []
        index = 0
        while len(seleted_words) < min(len(text_words_scores), label_num) and index < len(text_words_scores):
            token = text_words_scores[index][0]
            frequency = text_words_scores[index][1]
            index += 1
            if frequency > 0 and token not in seleted_words:
                seleted_words.append(token)
        train_selected_words.append({kb_id: seleted_words})
        text_tokens = []
        label = []
        for word in text_words:
            word_tokens = tokenizer.tokenize(word)
            text_tokens += word_tokens
            label += [1 for _ in range(len(word_tokens))] if word in seleted_words else \
                [0 for _ in range(len(word_tokens))]

        text_input_tokens.append(text_tokens)
        labels.append(label)
    return text_input_tokens, labels


def clear_tokens(tokens):
    tokens = [token for token in tokens if token not in REMOVED_TOKENS]
    return tokens


def clear_tokens_electra(tokens):
    tokens = [token for token in tokens if token not in REMOVED_ELECTRA_TOKENS]
    return tokens


def process_test_kb(test_kb, tokenizer, window_size, dtype):
    sequence_and_id = []

    for key, value in tqdm(test_kb.items()):
        description = remove_punctuation(value["text"]).lower()
        des_tokens = tokenizer.tokenize(description)
        des_tokens = clear_tokens_electra(des_tokens)
        sequence_and_id += [(des_tokens, key)]

    return sequence_and_id


class TrainDataset(Dataset):
    def __init__(self, samples_train_tokens, train_labels, tokenizer):
        super(TrainDataset, self).__init__()
        self.samples_train_tokens = samples_train_tokens
        self.train_labels = train_labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples_train_tokens)

    def __getitem__(self, index):
        sample_train_tokens = [self.tokenizer.cls_token] + self.samples_train_tokens[index] + \
                              [self.tokenizer.sep_token]
        train_label = [0] + self.train_labels[index] + [0]

        input_ids = self.tokenizer.convert_tokens_to_ids(sample_train_tokens)
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) <= 514
        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        labels = torch.tensor(train_label).float()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def train_allocate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    input_ids = pad_sequence(input_ids, True, 1)
    attention_mask = pad_sequence(attention_mask, True, 0)
    labels = pad_sequence(labels, True, -100.0)

    return input_ids, attention_mask, labels


class TestDataset(Dataset):
    def __init__(self, samples_test, tokenizer, max_text_len):
        super(TestDataset, self).__init__()
        self.samples_test = samples_test
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.samples_test)

    def __getitem__(self, index):
        sample_test = self.samples_test[index]
        text = sample_test["text"].lower()
        kb_id = sample_test["kb_id"]

        text_words = text.split(" ")
        start, end = text_words.index("[e1]"), text_words.index("[\e1]")
        mention_words = text_words[start + 1:end]
        mention_tokens = self.tokenizer.tokenize(" ".join(mention_words))
        half_text_len = (self.max_text_len - end + start) // 2
        text_words = text_words[max(0, start - half_text_len):end + half_text_len]
        input_tokens = []
        token_mark = []
        for i, word in enumerate(text_words):
            word_tokens = self.tokenizer.tokenize(word)
            input_tokens += word_tokens
            token_mark += [i for _ in range(len(word_tokens))]
        assert len(input_tokens) <= 512
        input_tokens = [self.tokenizer.cls_token] + input_tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)
        input_ids = torch.tensor(input_ids).long()
        attention_mask = torch.tensor(attention_mask).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": kb_id,
            "text_words": text_words,
            "token_mark": token_mark,
            "mention_tokens": mention_tokens
        }


def get_description_values(raw_kb, predicts, tokenizer, max_len, args):
    input_ids = []
    attention_masks = []
    for predict in predicts:
        text = raw_kb[predict]["text"]
        text = remove_punctuation(text)
        text = " ".join(text.split()[:max_len])
        text_tokens = tokenizer.tokenize(text)
        text_input_tokens = [tokenizer.cls_token] + text_tokens[:max_len - 2] + \
                            [tokenizer.sep_token]
        text_input_ids = tokenizer.convert_tokens_to_ids(text_input_tokens)
        input_ids.append(torch.tensor(text_input_ids).long())
        text_attention_masks = [1 for _ in range(len(text_input_ids))]
        attention_masks.append(torch.tensor(text_attention_masks).long())
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return input_ids, attention_masks


def get_train_dataset(samples_train_tokens, train_labels, tokenizer):
    return TrainDataset(samples_train_tokens, train_labels, tokenizer)


def get_test_dataset(samples_test, tokenizer, max_text_len):
    return TestDataset(samples_test, tokenizer, max_text_len)


def get_train_mention_loader(samples_train_tokens, train_labels, tokenizer, args):
    dataset = get_train_dataset(samples_train_tokens, train_labels, tokenizer)
    return DataLoader(dataset, args.B, shuffle=True, collate_fn=train_allocate_fn)


def get_test_mention_loader(samples_test, tokenizer, max_text_len):
    dataset = get_test_dataset(samples_test, tokenizer, max_text_len)
    return DataLoader(dataset, batch_size=1, shuffle=False)
