#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


import codecs
import json
import os
import numpy as np
import torch
import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

os.environ["http_proxy"] = "172.18.166.31:7899"
os.environ["https_proxy"] = "172.18.166.31:7899"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def contriever_embedding(texts):
    """embedding
    """
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever').to(device)

    embedding_list = []
    for t_i in tqdm(texts):
        texts = [t_i]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        embed_result = embeddings.cpu().tolist()[0]
        embedding_list.append(embed_result)
    
    return embedding_list


def load_dataset(data_file):
    """load dataset
    """
    dataset = []
    with codecs.open(data_file, "r", "utf-8") as f:
        for line in f:
            json_obj = json.loads(line.strip())
            dataset.append(json_obj)

    print(f"dataset size: {len(dataset)}")
    print(dataset[0].keys())
    print(dataset[0]["caseID"])
    print(dataset[0]["fact_split"]) # zhuguan, keguan, shiwai

    data_preprocess = []
    article_meta_info = dict()
    accusation_meta_info = dict()
    imprisonment_meta_info = dict()
    map_idx_caseid = dict()
    cnt_idx = 0
    for data_row in dataset:
        _case_id = data_row["caseID"]
        map_idx_caseid[cnt_idx] = _case_id

        _article = int(data_row["meta"]["relevant_articles"][0])
        if _article not in article_meta_info:
            article_meta_info[_article] = []
        article_meta_info[_article].append(cnt_idx)

        _accusation = data_row["meta"]["accusation"][0]
        if _accusation not in accusation_meta_info:
            accusation_meta_info[_accusation] = []
        accusation_meta_info[_accusation].append(cnt_idx)

        _imprisonment_tmp = data_row["meta"]["term_of_imprisonment"]
        _imprisonment = pt_cls2str[get_pt_cls(_imprisonment_tmp["imprisonment"])]
        if _imprisonment not in imprisonment_meta_info:
            imprisonment_meta_info[_imprisonment] = []
        imprisonment_meta_info[_imprisonment].append(cnt_idx)

        _zhuguan = data_row["fact_split"]["zhuguan"]
        _keguan = data_row["fact_split"]["keguan"]
        _shiwai = data_row["fact_split"]["shiwai"]
        data_preprocess.append({
            "caseID": _case_id,
            "zhuguan": _zhuguan,
            "keguan": _keguan,
            "shiwai": _shiwai,
            "text": f"{_zhuguan} || {_keguan} || {_shiwai}"
        })

        cnt_idx += 1

    meta_info = {
        "article": article_meta_info,
        "accusation": accusation_meta_info,
        "imprisonment": imprisonment_meta_info
    }

    return data_preprocess, meta_info, map_idx_caseid


def load_candidate_result(data_file, data_type):
    """load candidate result
    """
    if data_type == "article":
        _size = 3
    else:
        _size = 10
    with codecs.open(data_file, "r", "utf-8") as f:
        candidate_sort = json.load(f)
    candidate_sort = [x[:_size] for x in candidate_sort]
    return candidate_sort


def run_main(opts):
    """run main
    """
    _data_size = opts.data_size

    precedent_preprocess, precedent_meta_info, precedent_map_idx_caseid = load_dataset(data_file="../precedent_database/precedent_case_fact_split.json")
    
    texts = [x["text"] for x in precedent_preprocess]
    precedent_embeddings = contriever_embedding(texts=texts)
    precedent_embeddings = [np.array(x_i) for x_i in precedent_embeddings]

    test_preprocess, _, _ = load_dataset(data_file="../testset/cjo22/testset_fact_split.json")
    texts = [x["text"] for x in test_preprocess]
    test_embeddings = contriever_embedding(texts=texts)
    test_article_candidate = load_candidate_result(data_file=f"../output/domain_model_out/candidate_label/cjo22/{opts.model}/article_topk_{_data_size}.json", data_type="article")
    test_accusation_candidate = load_candidate_result(data_file=f"../output/domain_model_out/candidate_label/cjo22/{opts.model}/charge_topk_{_data_size}.json", data_type="accusation")
    test_imprisonment_candidate = load_candidate_result(data_file=f"../output/domain_model_out/candidate_label/cjo22/{opts.model}/penalty_topk_{_data_size}.json", data_type="imprisonment")

    article_retrieve_total = []
    accusation_retrieve_total = []
    imprisonment_retrieve_total = []

    for i, test_embed_i in tqdm(enumerate(test_embeddings)):
        
        # 相似度计算
        tensor_1 = [np.array(test_embed_i)]
        similarity = cosine_similarity(tensor_1, precedent_embeddings)[0]
        sim_arg = np.argsort(-similarity).tolist() # 按照相似度从大到小排列
        
        article_retrieve = []
        article_candidate_i = test_article_candidate[i] # 第一个testcase的候选article_id
        for ac_i in article_candidate_i:
            ac_i = int(ac_i)
            for sa_i in sim_arg:
                if sa_i in precedent_meta_info["article"].get(ac_i, []):
                    article_retrieve.append(sa_i)
                    break
        
        accusation_retrieve = []
        accusation_candidate_i = test_accusation_candidate[i]
        for cc_i in accusation_candidate_i:
            for sa_i in sim_arg:
                if sa_i in precedent_meta_info["accusation"][cc_i]:
                    accusation_retrieve.append(sa_i)
                    break

        imprisonment_retrieve = []
        imprisonment_candidate_i = test_imprisonment_candidate[i]
        for ic_i in imprisonment_candidate_i:
            for sa_i in sim_arg:
                if sa_i in precedent_meta_info["imprisonment"][ic_i]:
                    imprisonment_retrieve.append(sa_i)
                    break
        
        article_retrieve = [precedent_map_idx_caseid[x] for x in article_retrieve]
        article_retrieve_total.append(article_retrieve)

        accusation_retrieve = [precedent_map_idx_caseid[x] for x in accusation_retrieve]
        accusation_retrieve_total.append(accusation_retrieve)

        imprisonment_retrieve = [precedent_map_idx_caseid[x] for x in imprisonment_retrieve]
        imprisonment_retrieve_total.append(imprisonment_retrieve)

    # output
    output_dir = "../output/domain_model_out/precedent_idx/cjo22/dense_retrieval_cnn"
    with codecs.open(os.path.join(output_dir, f"precedent_idxs_article_{_data_size}.json"), "w", "utf-8") as f:
        json.dump(article_retrieve_total, f)

    with codecs.open(os.path.join(output_dir, f"precedent_idxs_charge_{_data_size}.json"), "w", "utf-8") as f:
        json.dump(accusation_retrieve_total, f)

    with codecs.open(os.path.join(output_dir, f"precedent_idxs_penalty_{_data_size}.json"), "w", "utf-8") as f:
        json.dump(imprisonment_retrieve_total, f)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="BERT", help="CNN, TextCNN or BERT")
    parser.add_argument("--data_size", type=str, default="6w", help="6w, 20w, 50w training dataset")

    opts = parser.parse_args()

    print(opts.model)
    print(opts.data_size)

    run_main(opts)