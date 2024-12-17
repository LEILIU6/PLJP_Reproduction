#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import time
import os
import json
import codecs
import torch
import argparse

from tqdm import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score

# BERT_VERSION = "bert-base-uncased"
BERT_VERSION = "bert-base-chinese"

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


def load_article_dataset(dataset_file, label2idx, origin=False, cjo22=False):
    """load article dataset
    """
    train_data = {
        "text": [],
        "label": []
    }
    with codecs.open(dataset_file, "r", "utf-8") as f:
        for line in f:
            json_obj = json.loads(line.strip())
            _fact = json_obj["fact"]
            if cjo22:
                _fact = "".join(_fact.split(" "))
            if origin:
                _term_of_key = label2idx[str(json_obj["meta"]["relevant_articles"][0])]
            else:
                _term_of_key = label2idx[str(json_obj["relevant_articles"][0])]
            train_data["text"].append(_fact)
            train_data["label"].append(_term_of_key)
    train_data = Dataset.from_dict(train_data)
    return train_data

def load_accusation_dataset(dataset_file, label2idx, origin=False):
    """load accusation dataset
    """
    train_data = {
        "text": [],
        "label": []
    }
    with codecs.open(dataset_file, "r", "utf-8") as f:
        for line in f:
            json_obj = json.loads(line.strip())
            _fact = json_obj["fact"]
            if origin:
                _term_of_key = label2idx[json_obj["meta"]["accusation"][0]]
            else:
                _term_of_key = label2idx[json_obj["accusation"][0]]
            train_data["text"].append(_fact)
            train_data["label"].append(_term_of_key)
    train_data = Dataset.from_dict(train_data)
    return train_data

def load_imprisonment_dataset(dataset_file, label2idx, origin=False):
    """load imprisonment dataset
    """
    train_data = {
        "text": [],
        "label": []
    }
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
            train_data["text"].append(_fact)
            train_data["label"].append(_term_of_key)
    train_data = Dataset.from_dict(train_data)
    return train_data

def preprocess_imprisonment_data(tokenizer, examples):
    # 将文本转为BERT输入格式
    encoding = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    encoding["labels"] = examples["label"]
    return encoding

def train(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for step_i, batch in tqdm(enumerate(train_dataloader)):
        input_ids = torch.stack(batch["input_ids"], dim=-1).to(device)
        token_type_ids = torch.stack(batch["token_type_ids"], dim=-1).to(device)
        attention_mask = torch.stack(batch["attention_mask"], dim=-1).to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
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
            token_type_ids = torch.stack(batch["token_type_ids"], dim=-1).to(device)
            attention_mask = torch.stack(batch["attention_mask"], dim=-1).to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

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
        train_dataset = load_article_dataset(dataset_file=f"dataset/{opts.run_type}_{opts.data_size}/train.json", label2idx=label2idx)
        test_dataset = load_article_dataset(dataset_file=test_dataset_file, label2idx=label2idx)
    elif run_type == "accusation":
        accusation_label_file = "../output/meta/c2i.json"
        with codecs.open(accusation_label_file, "r", "utf-8") as f:
            label2idx = json.load(f)
        train_dataset = load_accusation_dataset(dataset_file=f"dataset/{opts.run_type}_{opts.data_size}/train.json", label2idx=label2idx)
        test_dataset = load_accusation_dataset(dataset_file=test_dataset_file, label2idx=label2idx)
    elif run_type == "imprisonment":
        imprisonment_label_file = "../output/meta/p2i.json"
        with codecs.open(imprisonment_label_file, "r", "utf-8") as f:
            label2idx = json.load(f)
        train_dataset = load_imprisonment_dataset(dataset_file=f"dataset/{opts.run_type}_{opts.data_size}/train.json", label2idx=label2idx)
        test_dataset = load_imprisonment_dataset(dataset_file=test_dataset_file, label2idx=label2idx)

    tokenizer = BertTokenizer.from_pretrained(BERT_VERSION)
    model = BertForSequenceClassification.from_pretrained(BERT_VERSION, num_labels=len(label2idx))  # 假设是二分类任务
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=opts.lr)

    train_dataset = train_dataset.map(lambda x: preprocess_imprisonment_data(tokenizer, x), batched=True)
    train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)
    test_dataset = test_dataset.map(lambda x: preprocess_imprisonment_data(tokenizer, x), batched=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size)

    epoches = opts.epoches
    accuracy_list = []
    max_accuracy = 0.
    _test_dataset_desc = f"{test_dataset_name}_{run_type}"
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
            output_dir = f"./{opts.run_type}_{opts.data_size}_BERTC_saved_model_hyperparam"
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
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
        test_dataset = load_article_dataset(dataset_file=test_data_file, label2idx=label2idx, origin=origin)
    elif run_type == "accusation":
        accusation_label_file = "../output/meta/c2i.json"
        with codecs.open(accusation_label_file, "r", "utf-8") as f:
            label2idx = json.load(f)
        test_dataset = load_accusation_dataset(dataset_file=test_data_file, label2idx=label2idx, origin=origin)
    elif run_type == "imprisonment":
        imprisonment_label_file = "../output/meta/p2i.json"
        with codecs.open(imprisonment_label_file, "r", "utf-8") as f:
            label2idx = json.load(f)
        test_dataset = load_imprisonment_dataset(dataset_file=test_data_file, label2idx=label2idx, origin=origin)

    print(test_dataset[0])

    idx2label = dict([(v, k) for k, v in label2idx.items()])

    tokenizer = BertTokenizer.from_pretrained(BERT_VERSION)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    test_dataset = test_dataset.map(lambda x: preprocess_imprisonment_data(tokenizer, x), batched=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size)

    if cjo22_flag:
        dataset_name = f"{run_type}_cjo22"
    else:
        dataset_name = f"{run_type}_cail2018"
    print(f"dataset_name: {dataset_name}")
    accuracy, logits_list, labels_list  = evaluate(model, test_dataloader, device, dataset_name)
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
        with codecs.open(f"../output/domain_model_out/candidate_label/cjo22/BERT/{out_type}_topk_{opts.data_size}.json", "w", "utf-8") as f:
            json.dump(topk_label, f, ensure_ascii=False)


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_type", type=str, default="imprisonment", help="article, accusation, imprisonment")
    parser.add_argument("--data_size", type=str, default="6w", help="6w, 20w, 50w training dataset")
    parser.add_argument("--run_train", action="store_true")
    parser.add_argument("--run_infer", action="store_true")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)

    opts = parser.parse_args()

    if opts.run_train:
        run_type = opts.run_type
        run_train(run_type, opts=opts)

    if opts.run_infer:
        run_type = opts.run_type
        data_size = opts.data_size
        model_dir=f"./{run_type}_{data_size}_BERTC_saved_model"
        save_result=False
        run_infer(run_type, model_dir, save_result=save_result, opts=opts)
