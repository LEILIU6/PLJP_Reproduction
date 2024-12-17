# -*- coding: utf-8 -*-

import openai
import os
import time
import json
import threading
import time
from tqdm import tqdm
import argparse
from mycode.utils import loader
from mycode.utils import prompt_gen as prompt_gen
import tiktoken
import requests
from copy import deepcopy

# url = "http://10.15.82.10:8000/v1/chat/completions"

# # 设置代理
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

key_pool = [
    ""
]
openai.api_key = key_pool[0]


def dav_response(text_list, model_name):
    enc = tiktoken.get_encoding("p50k_base")
    max_token = [len(enc.encode(t)) for t in text_list]
    max_token = 4096 - max(max_token)
    response = openai.Completion.create(
        model=model_name, prompt=text_list, max_tokens=max_token)  # greedy
    return response

def dav2_response(text_list):
    return dav_response(text_list, "text-davinci-002")

def dav3_response(text_list):
    return dav_response(text_list, "text-davinci-003")

def turbo_response(text_list):
    assert len(text_list) == 1, "gpt-3.5-turbo使用batch必须为1"
    text = text_list[0]
    enc = tiktoken.get_encoding("cl100k_base")
    max_token = 4096 - len(enc.encode(text)) - 500
    # (1608 in the messages, 2496 in the completion
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo", messages=[{"role": "user", "content": text}], max_tokens=max_token, temperature=0.01, top_p=1)
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": text}], max_tokens=max_token, temperature=0.01, top_p=1)
    return response

def mini4o_response(text_list):
    assert len(text_list) == 1, "gpt-3.5-turbo使用batch必须为1"
    text = text_list[0]
    enc = tiktoken.get_encoding("cl100k_base")
    max_token = 4096 - len(enc.encode(text)) - 500
    # (1608 in the messages, 2496 in the completion
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo", messages=[{"role": "user", "content": text}], max_tokens=max_token, temperature=0.01, top_p=1)
        model="gpt-4o-mini", messages=[{"role": "user", "content": text}], max_tokens=max_token, temperature=0.01, top_p=1)
    return response

def qwen_response(text_list):
    assert len(text_list) == 1, "qwen使用batch必须为1"
    text = text_list[0]

    url = "https://api.siliconflow.cn/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": text
            }
        ],
        "stream": False,
        "temperature": 0.01
    }
    siliconflow_key = ""
    headers = {
        "Authorization": f"Bearer {siliconflow_key}",
        "Content-Type": "application/json"
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    time.sleep(1)
    return response.json()

def get_openai_response(data, in_context_contents, key_pool, args, meta):
    assert len(data) == len(in_context_contents)
    llm_response_function = {
        "text-davinci-002": dav2_response,
        "text-davinci-003": dav3_response,
        "gpt-3.5-turbo": turbo_response,
        "gpt-4o-mini": mini4o_response,
        "qwen-2.5-32b": qwen_response
    }
    response_function = llm_response_function[args.model]

    truncated_data = []
    for i in tqdm(range(len(data)), desc="pre-process prompt"):
        case = data[i]
        fact = case["fact"].replace(" ", "") # raw fact
        caseID = case["caseID"]
        if args.use_split_fact: # reframed fact
            fact = [case["fact_split"]["zhuguan"], case["fact_split"]["keguan"], case["fact_split"]["shiwai"]]
            fact = "。".join(fact)

        max_len = 512
        fact = loader.truncate_text(fact, max_len=max_len)
        prompt = prompt_gen.retrieved_label_option_fewshot(fact, in_context_contents[i], args, caseID)

        obj = {}
        obj["prompt"] = prompt
        obj["caseID"] = case["caseID"]
        truncated_data.append(obj)
        # break

    print("Starting")
    i = 0
    batch = args.batch
    while i < len(truncated_data):
        print(f"=== {i} / {len(data)} ===")
        prompt_list = truncated_data[i:i+batch]
        text_list = [t["prompt"] for t in prompt_list]
        print(text_list[0]) # batch = 1
        try:
            response = response_function(text_list)
            resp_temp = deepcopy(response)
            responses = []
            for run in range(len(prompt_list)):
                resp_temp["choices"] = [response["choices"][run]]
                resp_temp["caseID"] = prompt_list[run]["caseID"]
                responses.append(deepcopy(resp_temp))
        except openai.error.RateLimitError as e:
            e = repr(e)
            print(e)
            if "limit" in e:
                time.sleep(60)
            elif "quota" in e:
                if len(key_pool) == 0:
                    print("用光了key！")
                    exit()
                print(f"=== 当前key: {key_pool[0]} ===")
                openai.api_key = key_pool[0]
                key_pool = key_pool[1:]
                time.sleep(1)
            continue
        except openai.error.APIError as e:
            e = repr(e)
            print(e)
            time.sleep(60)
        except Exception as e:
            # https://platform.openai.com/docs/guides/error-codes/api-errors
            e = repr(e)
            print(e)
            exit()

        time.sleep(0.5)

        with open(args.output_path, "a+", encoding="utf-8") as f:
            resp = responses[0]
            line_obj = {
                "llm_response": resp,
                "prompt": text_list[0]
            }
            line = json.dumps(line_obj, ensure_ascii=False)
            f.write(line + "\n")
        i += batch
        # break


def run_llm(args):
    data = []

    with open(args.input_path, encoding="utf-8") as f:
        for idx, line in enumerate(f.readlines()):
            case = json.loads(line)
            data.append(case)

    dumped_data = []
    with open(args.output_path, encoding="utf-8") as f:
        for line in f.readlines():
            dumped_data.append(line)
    
    print(f"dumped_data: {dumped_data}")

    data = data[len(dumped_data):]
    print("to run data count: ", len(data))

    # load precedents
    precedents = loader.load_precedent(args.precedent_pool_path, args.precedent_idx_path)
    precedents = precedents[len(dumped_data):]
    # #纯随机选择类案
    # precedents = [precedents[0]]*(len(data))

    # load [charge, article, penalty] topk prediction
    topk_label_option = loader.load_topk_option(args.topk_label_option_path)
    topk_label_option = topk_label_option[len(dumped_data):]

    # load relevant article definition
    if args.task == "article":
        article_definition = loader.load_retrieved_articles(args.topk_label_option_path, index_type="num")
        article_definition = article_definition[len(dumped_data):]
    else:
        article_definition = [["#"] for _ in range(len(data))]
    
    in_context_contents = [{
        "precedents": precedents[i],
        "topk_label_option": topk_label_option[i],
        "article_definition": article_definition[i], 
    }
        for i in range(len(precedents))]

    # # meta knowledge
    meta = {}
    # in_context_contents = [None]*len(data)
    get_openai_response(data, in_context_contents, key_pool, args, meta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # base setting
    parser.add_argument(
        '--model', default='text-davinci-003', help='text-davinci-002/3 & gpt-3.5-turbo')
    parser.add_argument(
        '--dataset', default="cjo22", help='[cail18, cjo22]')
    parser.add_argument(
        '--small_model', default="CNN", help='[CNN, TopJudge, ELE]')
    parser.add_argument(
        '--task', default="charge", help='[charge, article, penalty]')
    parser.add_argument(
        '--shot', default=3, help='few shots')
    parser.add_argument(
        '--batch', default=1, help='batch, reduce throughput cost')
    parser.add_argument(
        '--retriever', default="dense_retrieval", help="[bm25, dense_retrieval]")
    parser.add_argument(
        '--use_split_fact', default=False, help="[fact / fact_split]")
    
    # input: testset path, output: llm response path
    parser.add_argument(
        '--input_path', default="", help='testset path')
    parser.add_argument(
        '--output_path', default="", help='llm response path')
    
    # in-context contents path
    parser.add_argument(
        '--precedent_pool_path', default="", help='pool of precedent')
    parser.add_argument(
        '--precedent_idx_path', default="", help='index list of precedent')
    parser.add_argument(
        '--topk_label_option_path', default="", help='index list of topk label option')
    parser.add_argument(
        '--predicted_article_path', default="", help='index the predicted article to help predict charge and penalty')
    parser.add_argument(
        '--predicted_charge_path',default="",help='index the predicted charge to help predict penalty')
    parser.add_argument(
        '--train_size', default="6w", help='training size'
    )
    
    args = parser.parse_args()

    if args.input_path == "":
        prefix = f"data/testset/{args.dataset}/"
        if not args.use_split_fact:
            args.input_path = prefix + "testset.json"
        else:
            args.input_path = prefix + "testset_fact_split.json" 
        
    if args.output_path == "":
        args.output_path = f"./data/output/llm_out/{args.dataset}/{args.small_model}/{args.task}/{args.shot}shot/{args.model}_{args.train_size}.json"
        if not os.path.exists(args.output_path):
            dirname = os.path.dirname(args.output_path)
            os.makedirs(dirname, exist_ok=True)
            with open(args.output_path, "w") as f:
                pass

    if args.precedent_pool_path == "":
        prefix = f"data/precedent_database/"
        if not args.use_split_fact:
            args.precedent_pool_path = prefix + "precedent_case.json"
        else:
            args.precedent_pool_path = prefix + "precedent_case_fact_split.json" 
        
    if args.precedent_idx_path == "":
        prefix = f"data/output/domain_model_out/precedent_idx/{args.dataset}/{args.retriever}_{args.small_model.lower()}/"
        args.precedent_idx_path = prefix + f"precedent_idxs_{args.task}_{args.train_size}.json"


    if args.topk_label_option_path == "":
        args.topk_label_option_path = f"data/output/domain_model_out/candidate_label/{args.dataset}/{args.small_model}/{args.task}_topk_{args.train_size}.json"


    if args.predicted_article_path == "":
        args.predicted_article_path = f"data/output/llm_out/{args.dataset}/{args.small_model}/article/{args.shot}shot/{args.model}_{args.train_size}.json"

    if args.predicted_charge_path == "":
        args.predicted_charge_path = f"data/output/llm_out/{args.dataset}/{args.small_model}/charge/{args.shot}shot/{args.model}_{args.train_size}.json"

    start_time = time.time()
    run_llm(args)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.6f} seconds")
