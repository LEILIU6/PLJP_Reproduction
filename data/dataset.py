#!/usr/bin/env python3
# -*- encoding: utf-8 -*-



import json
import codecs
import argparse


def sample_dataset(opts):
    """sample dataset
    """
    _map_str_int = {
        "50w": 500000,
        "20w": 200000,
        "6w": 65711
    }
    _data_size = _map_str_int[opts.data_size]

    article_to_index_file = "output/meta/a2i.json"
    with codecs.open(article_to_index_file, "r", "utf-8") as f:
        a2i = json.load(f)

    accusation_to_index_file = "output/meta/c2i.json"
    with codecs.open(accusation_to_index_file, "r", "utf-8") as f:
        c2i = json.load(f)

    imprisonment_to_index_file = "output/meta/p2i.json"
    with codecs.open(imprisonment_to_index_file, "r", "utf-8") as f:
        p2i = json.load(f)

    # 加载过滤后的数据集, 构造[train/dev/test]

    dataset = []
    with codecs.open("predictive_model/dataset.json", "r", "utf-8") as f:
        for line in f:
            data_row = json.loads(line.strip())
            _relevant_articles = data_row["relevant_articles"]
            assert len(_relevant_articles) == 1
            assert _relevant_articles[0] in a2i.keys()
            _accusation = data_row["accusation"]
            assert len(_accusation) == 1
            assert _accusation[0] in c2i.keys()
            _term_of_imprisonment = data_row["term_of_imprisonment"]
            assert _term_of_imprisonment in p2i.keys()
            dataset.append(data_row)

    print(len(dataset))

    # 数据集的标签分布
    ## article
    article_dist = dict()
    accusation_dist = dict()
    imprisonment_dist = dict()
    for data_row in dataset:
        _article = data_row["relevant_articles"][0]
        _accusation = data_row["accusation"][0]
        _imprisonment = data_row["term_of_imprisonment"]

        if _article not in article_dist:
            article_dist[_article] = []
        article_dist[_article].append(data_row)

        if _accusation not in accusation_dist:
            accusation_dist[_accusation] = []
        accusation_dist[_accusation].append(data_row)

        if _imprisonment not in imprisonment_dist:
            imprisonment_dist[_imprisonment] = []
        imprisonment_dist[_imprisonment].append(data_row)
    article_dist = sorted(list(article_dist.items()), key=lambda x: len(x[1]), reverse=True)
    accusation_dist = sorted(list(accusation_dist.items()), key=lambda x: len(x[1]), reverse=True)
    imprisonment_dist = sorted(list(imprisonment_dist.items()), key=lambda x: len(x[1]), reverse=True)

    for _run_type in ["article", "accusation", "imprisonment"]:
        print(f">>> constructing {_run_type} data ...")
        # 根据标签分布抽样，优先保证训练集的数据分布
        smaller_dataset = {
            "train": [],
            "dev": [],
            "test": []
        }
        train_flag = True
        dev_flag = True
        test_flag = True

        if _run_type == "article":
            target_dict = article_dist
        elif _run_type == "accusation":
            target_dict = accusation_dist
        elif _run_type == "imprisonment":
            target_dict = imprisonment_dist
        else:
            raise ValueError("error in run_type")

        while True:
            for k, v in target_dict:
                if len(v) == 0:
                    continue
                if len(smaller_dataset["train"]) < _data_size:
                    smaller_dataset["train"].append(v.pop())
                else:
                    train_flag = False
                    break
            if not train_flag:
                break

        while True:
            for k, v in target_dict:
                if len(v) == 0:
                    continue
                if len(smaller_dataset["dev"]) < 8214:
                    smaller_dataset["dev"].append(v.pop())
                else:
                    dev_flag = False
                    break
            if not dev_flag:
                break

        while True:
            for k, v in target_dict:
                if len(v) == 0:
                    continue
                if len(smaller_dataset["test"]) < 8214:
                    smaller_dataset["test"].append(v.pop())
                else:
                    test_flag = False
                    break
            if not test_flag:
                break

        print(len(smaller_dataset["train"]))
        print(len(smaller_dataset["dev"]))
        print(len(smaller_dataset["test"]))

        with codecs.open(f"predictive_model/dataset/{_run_type}_{opts.data_size}/train.json", "w", "utf-8") as f:
            for data_row in smaller_dataset["train"]:
                f.write(json.dumps(data_row, ensure_ascii=False) + "\n")

        with codecs.open(f"predictive_model/dataset/{_run_type}_{opts.data_size}/dev.json", "w", "utf-8") as f:
            for data_row in smaller_dataset["dev"]:
                f.write(json.dumps(data_row, ensure_ascii=False) + "\n")

        with codecs.open(f"predictive_model/dataset/{_run_type}_{opts.data_size}/test.json", "w", "utf-8") as f:
            for data_row in smaller_dataset["test"]:
                f.write(json.dumps(data_row, ensure_ascii=False) + "\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_size", type=str, default="6w", help="6w, 20w, 50w training dataset")

    opts = parser.parse_args()

    sample_dataset(opts)