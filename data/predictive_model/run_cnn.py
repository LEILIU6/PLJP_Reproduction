#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json
import codecs
import torch
import argparse
import numpy as np

from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score


class CNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_classes, num_filters, dropout=0.5):
        super(CNN, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Convolution layers with multiple kernel sizes
        self.conv_layer = nn.Conv2d(1, num_filters, (5, embed_dim))
        # Fully connected layer
        self.fc = nn.Linear(num_filters, num_classes)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, seq_len)
        # Embedding lookup: (batch_size, seq_len, embed_dim)
        x = self.embedding(x)
        # Add a channel dimension: (batch_size, 1, seq_len, embed_dim)
        x = x.unsqueeze(1)
        # Convolution + ReLU + Max-pooling
        # Convolution output: (batch_size, num_filters, seq_len - kernel_size + 1, 1)
        conv_out = F.relu(self.conv_layer(x))
        # Max-pooling over the sequence dimension
        pooled_out = F.max_pool2d(conv_out, (conv_out.shape[2], 1)).squeeze(3).squeeze(2)
        # Apply dropout
        x = self.dropout(pooled_out)
        # Fully connected layer: (batch_size, num_classes)
        logits = self.fc(x)
        return logits

class TextCNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes, num_filters, dropout=0.5):
        super(TextCNN, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Convolution layers with multiple kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        # Fully connected layer
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (batch_size, seq_len)
        # Embedding lookup: (batch_size, seq_len, embed_dim)
        x = self.embedding(x)
        # Add a channel dimension: (batch_size, 1, seq_len, embed_dim)
        x = x.unsqueeze(1)
        # Convolution + ReLU + Max-pooling for each kernel size
        conv_outputs = []
        for conv in self.conv_layers:
            # Convolution output: (batch_size, num_filters, seq_len - kernel_size + 1, 1)
            conv_out = F.relu(conv(x))
            # Max-pooling over the sequence dimension
            pooled_out = F.max_pool2d(conv_out, (conv_out.shape[2], 1)).squeeze(3).squeeze(2)
            conv_outputs.append(pooled_out)
        # Concatenate outputs from all kernel sizes: (batch_size, num_filters * len(kernel_sizes))
        x = torch.cat(conv_outputs, dim=1)
        # Apply dropout
        x = self.dropout(x)
        # Fully connected layer: (batch_size, num_classes)
        logits = self.fc(x)
        return logits
    
pt_cls2str = ["其他", "六个月以下", "六到九个月", "九个月到一年", "一到两年", "二到三年", "三到五年", "五到七年", "七到十年", "十年以上"]

def get_pt_cls(pt):
    if pt > 10 * 12:
        pt_cls = 9
    elif pt > 7 * 12:
        pt_cls = 8
    elif pt > 5 * 12:
        pt_cls = 7
    elif pt > 3 * 12:
        pt_cls = 6
    elif pt > 2 * 12:
        pt_cls = 5
    elif pt > 1 * 12:
        pt_cls = 4
    elif pt > 9:
        pt_cls = 3
    elif pt > 6:
        pt_cls = 2
    elif pt > 0:
        pt_cls = 1
    else:
        pt_cls = 0
    return pt_cls


def load_article_dataset(dataset_file, label2idx, origin=False, cjo22=False, train=False):
    """load article dataset
    """
    train_data = {
        "text": [],
        "label": []
    }
    w2i = {"<PAD>": 0} if train else None
    with codecs.open(dataset_file, "r", "utf-8") as f:
        for line in f:
            json_obj = json.loads(line.strip())
            _fact = json_obj["fact"].strip()
            if cjo22:
                _fact = "".join(_fact.split(" "))
            if origin:
                _term_of_key = label2idx[str(json_obj["meta"]["relevant_articles"][0])]
            else:
                _term_of_key = label2idx[str(json_obj["relevant_articles"][0])]
            if train:
                # tokenization
                _fact_tokens = " ".join(_fact).split(" ")
                for _tk_i in _fact_tokens:
                    if _tk_i not in w2i:
                        w2i[_tk_i] = len(w2i)
            train_data["text"].append(_fact)
            train_data["label"].append(_term_of_key)
    train_data = Dataset.from_dict(train_data)
    return train_data, w2i


def load_accusation_dataset(dataset_file, label2idx, origin=False, train=False):
    """load accusation dataset
    """
    train_data = {
        "text": [],
        "label": []
    }
    w2i = {"<PAD>": 0} if train else None
    with codecs.open(dataset_file, "r", "utf-8") as f:
        for line in f:
            json_obj = json.loads(line.strip())
            _fact = json_obj["fact"]
            if origin:
                _term_of_key = label2idx[json_obj["meta"]["accusation"][0]]
            else:
                _term_of_key = label2idx[json_obj["accusation"][0]]
            if train:
                # tokenization
                _fact_tokens = " ".join(_fact).split(" ")
                for _tk_i in _fact_tokens:
                    if _tk_i not in w2i:
                        w2i[_tk_i] = len(w2i)
            train_data["text"].append(_fact)
            train_data["label"].append(_term_of_key)
    train_data = Dataset.from_dict(train_data)
    return train_data, w2i

def load_imprisonment_dataset(dataset_file, label2idx, origin=False, train=False):
    """load imprisonment dataset
    """
    train_data = {
        "text": [],
        "label": []
    }
    w2i = {"<PAD>": 0} if train else None
    with codecs.open(dataset_file, "r", "utf-8") as f:
        for line in f:
            json_obj = json.loads(line.strip())
            _fact = json_obj["fact"]
            if origin:
                _imprisonment_tmp = json_obj["meta"]["term_of_imprisonment"]
                _imprisonment = pt_cls2str[get_pt_cls(_imprisonment_tmp["imprisonment"])]
                _term_of_key = label2idx[_imprisonment]
            else:
                _term_of_key = label2idx[json_obj["term_of_imprisonment"]]
            if train:
                # tokenization
                _fact_tokens = " ".join(_fact).split(" ")
                for _tk_i in _fact_tokens:
                    if _tk_i not in w2i:
                        w2i[_tk_i] = len(w2i)
            train_data["text"].append(_fact)
            train_data["label"].append(_term_of_key)
    train_data = Dataset.from_dict(train_data)
    return train_data, w2i

def train(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    for step_i, batch in tqdm(enumerate(train_dataloader)):
        input_ids = torch.stack(batch["input_ids"], dim=-1).to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step_i % 50 == 0:
            print(f">>> step_{step_i}, train loss: {loss}")
    avg_loss = total_loss / len(train_dataloader)
    return avg_loss

def evaluate(model, test_dataloader, device, eval_dataset_name):
    model.eval()
    preds_list, labels_list, logits_list = [], [], []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = torch.stack(batch["input_ids"], dim=-1).to(device)
            logits = model(input_ids)

            if eval_dataset_name == "imprisonment_cjo22":
                # 因为cjo22 imprisonment ‘0’标签
                _batch_preds_list = []
                _arg_sort = torch.argsort(-logits, dim=-1).cpu().tolist() # 从小到大排列
                for as_i in _arg_sort:
                    if as_i[0] == 0:
                        _batch_preds_list.append(as_i[1])
                    else:
                        _batch_preds_list.append(as_i[0])
                preds_list.extend(_batch_preds_list)
            else:
                preds_list.extend(torch.argmax(logits, dim=-1).cpu().tolist())

            labels_list.extend(batch["labels"].cpu().tolist())
            logits_list.extend(logits.cpu().tolist())
    accuracy = accuracy_score(labels_list, preds_list)
    return accuracy, logits_list, labels_list

def preprocess_data(w2i, examples):
    """preprocess data
    """
    encoding = {
        "input_ids": None,
        "labels": None
    }
    pad_size = 256
    text_token = " ".join(examples["text"]).split(" ")
    input_ids = [w2i.get(x_i, 0) for x_i in text_token]
    while len(input_ids) < pad_size:
        input_ids.append(0)
    input_ids = input_ids[:pad_size]
    encoding["input_ids"] = input_ids
    encoding["labels"] = examples["label"]
    return encoding

def run_train(run_type, opts):
    """main
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset_name = "cjo22"
    test_dataset_file = f"../testset/{test_dataset_name}/testset_transform.json"

    if run_type == "article":
        article_label_file = "../output/meta/a2i.json"
        with codecs.open(article_label_file, "r", "utf-8") as f:
            label2idx = json.load(f)
        train_dataset, w2i = load_article_dataset(dataset_file=f"dataset/{opts.run_type}_{opts.data_size}/train.json", label2idx=label2idx, train=True)
        test_dataset, _ = load_article_dataset(dataset_file=test_dataset_file, label2idx=label2idx)
    elif run_type == "accusation":
        accusation_label_file = "../output/meta/c2i.json"
        with codecs.open(accusation_label_file, "r", "utf-8") as f:
            label2idx = json.load(f)
        train_dataset, w2i = load_accusation_dataset(dataset_file=f"dataset/{opts.run_type}_{opts.data_size}/train.json", label2idx=label2idx, train=True)
        test_dataset, _ = load_accusation_dataset(dataset_file=test_dataset_file, label2idx=label2idx)
    elif run_type == "imprisonment":
        imprisonment_label_file = "../output/meta/p2i.json"
        with codecs.open(imprisonment_label_file, "r", "utf-8") as f:
            label2idx = json.load(f)
        train_dataset, w2i = load_imprisonment_dataset(dataset_file=f"dataset/{opts.run_type}_{opts.data_size}/train.json", label2idx=label2idx, train=True)
        test_dataset, _ = load_imprisonment_dataset(dataset_file=test_dataset_file, label2idx=label2idx)

    vocab_size = len(w2i)
    num_classes = len(label2idx)

    if opts.model_name == "CNN":
        model = CNN(vocab_size=vocab_size, embed_dim=300, num_classes=num_classes, num_filters=128, dropout=0.5)
    elif opts.model_name == "TextCNN":
        model = TextCNN(vocab_size=vocab_size, embed_dim=300, num_classes=num_classes, kernel_sizes=[3, 4, 5], num_filters=128, dropout=0.5)
    else:
        raise ValueError(f"invalid opts.model_name: {opts.model_name}")
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opts.lr)

    train_dataset = train_dataset.map(lambda x: preprocess_data(w2i, x))
    train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
    test_dataset = test_dataset.map(lambda x: preprocess_data(w2i, x))
    test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size)

    epoches = opts.epoches
    accuracy_list = []
    max_accuracy = 0.
    _test_dataset_desc = f"{run_type}_{test_dataset_name}" # _cjo22
    for epoch in range(epoches):
        start_time = time.time()
        avg_train_loss = train(model, train_dataloader, optimizer, device)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime: {runtime:.6f} seconds")
        accuracy, _, _ = evaluate(model, test_dataloader, device, _test_dataset_desc)
        print(f"Epoch {epoch+1}/{epoches} - Average Training Loss: {avg_train_loss:.4f} - Test Accuracy: {accuracy:.4f}")
        accuracy_list.append(accuracy)
        if accuracy >= max_accuracy:
            output_dir = f"{opts.run_type}_{opts.data_size}_{opts.model_name.lower}_saved_model"
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            model_file = os.path.join(output_dir, "model.pth")
            torch.save(model.state_dict(), model_file)
            w2i_file = os.path.join(output_dir, "w2i.json")
            with codecs.open(w2i_file, "w", "utf-8") as f:
                json.dump(w2i, f, indent=4, ensure_ascii=False)
            max_accuracy = accuracy
        
        print(accuracy_list)
        print(max_accuracy)

def run_infer(run_type, model_dir, save_result, opts):
    """run infer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data_file, origin, cjo22_flag = "../testset/cjo22/testset_transform.json", False, True
    print(f">>> running test dataset: {test_data_file}")

    if run_type == "article":
        article_label_file = "../output/meta/a2i.json"
        with codecs.open(article_label_file, "r", "utf-8") as f:
            label2idx = json.load(f)
        test_dataset, _ = load_article_dataset(dataset_file=test_data_file, label2idx=label2idx, origin=origin)
    elif run_type == "accusation":
        accusation_label_file = "../output/meta/c2i.json"
        with codecs.open(accusation_label_file, "r", "utf-8") as f:
            label2idx = json.load(f)
        test_dataset, _ = load_accusation_dataset(dataset_file=test_data_file, label2idx=label2idx, origin=origin)
    elif run_type == "imprisonment":
        imprisonment_label_file = "../output/meta/p2i.json"
        with codecs.open(imprisonment_label_file, "r", "utf-8") as f:
            label2idx = json.load(f)
        test_dataset, _ = load_imprisonment_dataset(dataset_file=test_data_file, label2idx=label2idx, origin=origin)

    print(test_dataset[0])
    w2i = json.load(codecs.open(os.path.join(model_dir, "w2i.json")))
    vocab_size = len(w2i)
    num_classes = len(label2idx)

    idx2label = dict([(v, k) for k, v in label2idx.items()])

    if opts.model_name == "CNN":
        model = CNN(vocab_size=vocab_size, embed_dim=300, num_classes=num_classes, num_filters=128, dropout=0.5)
    elif opts.model_name == "TextCNN":
        model = TextCNN(vocab_size=vocab_size, embed_dim=300, num_classes=num_classes, kernel_sizes=[3, 4, 5], num_filters=128, dropout=0.5)
    else:
        raise ValueError(f"invalid opts.model_name: {opts.model_name}")
    
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    model.to(device)

    test_dataset = test_dataset.map(lambda x: preprocess_data(w2i, x))
    test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size)

    if cjo22_flag:
        dataset_name = f"{run_type}_cjo22"
    else:
        dataset_name = f"{run_type}_cail2018"
    print(f"dataset_name: {dataset_name}")
    accuracy, logits_list, _  = evaluate(model, test_dataloader, device, dataset_name)
    print(f"Test Accuracy: {accuracy:.4f}")

    logits_list = torch.tensor(logits_list) # tensor
    logits_sort_arg = torch.argsort(-logits_list, dim=-1).tolist()
    topk_label = [[idx2label[y_i] for y_i in x_i] for x_i in logits_sort_arg]
    
    if run_type == "article":
        out_type = "article"
    elif run_type == "accusation":
        out_type = "charge"
    elif run_type == "imprisonment":
        out_type = "penalty"

    if save_result:
        with codecs.open(f"../output/domain_model_out/candidate_label/cjo22/{opts.model_name}/{out_type}_topk_{opts.data_size}.json", "w", "utf-8") as f:
            json.dump(topk_label, f, ensure_ascii=False)
    
if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_type", type=str, default="imprisonment", help="article, accusation, imprisonment")
    parser.add_argument("--data_size", type=str, default="6w", help="6w, 20w, 50w training dataset")
    parser.add_argument("--run_train", action="store_true")
    parser.add_argument("--run_infer", action="store_true")
    parser.add_argument("--model_name", type=str, default="CNN", help="CNN, TextCNN")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)

    opts = parser.parse_args()

    if opts.run_train:
        run_type = opts.run_type
        run_train(run_type, opts=opts)

    if opts.run_infer:
        run_type = opts.run_type
        data_size = opts.data_size
        model_dir=f"./{run_type}_{data_size}_{opts.model_name}_saved_model"
        save_result=True
        run_infer(run_type, model_dir, save_result=save_result, opts=opts)