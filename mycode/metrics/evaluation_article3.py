
import codecs
import json
import re
import cn2an
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import jaccard_score

data = []

testset_path = "data/testset/cjo22/testset.json"

with open(testset_path, encoding="utf8") as f:
    for line in f.readlines():
        obj = json.loads(line)
        data.append(obj)

results = []
resp_file = "data/output/llm_out/cjo22/BERTC/article/3shot/qwen-2.5-32b_6w.json"

with open(resp_file, encoding="utf8") as f:
    for i, line in enumerate(f.readlines()):
        obj = json.loads(line)
        results.append(obj)

y_true = []
y_true_all = []
for case in data:
    ar = case["meta"]["relevant_articles"]
    ar = [int(x) for x in ar]
    y_true_all.append(ar)
    y_true.append(ar[0])


y_pred = []
count = 0
fail_count = 0
for i, obj in enumerate(results):
    count += 1
    # resp = obj["choices"][0]["text"] # dav
    try:
        resp = obj["llm_response"]["choices"][0]["message"]['content'] # turbo
        json_match = re.search(r"\{.*\}", resp, re.DOTALL)
        extracted_json = json_match.group()
        matched_text = json.loads(extracted_json)["相关法条"]

        # match = re.search(r"'相关法条':\s*'第\d+条", resp)
        # matched_text = match.group() if match else None

        res = re.findall(r'\d+', matched_text)
        ars = []
        for i in res:
            if i.isdigit():
                i = int(i)
                ars.append(i)
            else:
                try:
                    i = cn2an.cn2an(i, mode="smart")
                    ars.append(i)
                except:
                    ars.append(0)
        if len(ars) == 0:
            y_pred.append(0)
            print(i)
        else:
            y_pred.append(ars[0])
    except BaseException as e:
        print(e)
        print(obj["llm_response"]["choices"][0]["message"]['content'])
        y_pred.append(0)
        fail_count += 1


badcase_list = []
for i, r_i in enumerate(results):
    _badcase = {
        "prompt": r_i["prompt"],
        "llm_response": r_i["llm_response"]["choices"][0]["message"]['content']
    }
    yt_i = y_true[i]
    yta_i = y_true_all[i]
    yp_i = y_pred[i]
    if yt_i != yp_i:
        _badcase["y_pred"] = yp_i
        _badcase["y_true"] = yt_i
        _badcase["y_true_all"] = yta_i
        badcase_list.append(_badcase)

with codecs.open("article_badcase.json", "w", "utf-8") as f:
    json.dump(badcase_list, f, indent=4, ensure_ascii=False)
    
acc, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
map, mar, maf, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

acc = round(acc, 6) * 100
map = round(map, 6) * 100
mar = round(mar, 6) * 100
maf = round(maf, 6) * 100

print(f"acc:{acc}, map:{map}, mar:{mar}, maf:{maf}")


print(len(y_pred))
print(y_pred[-10:])
print(len(y_true))
print(y_true[-10:])

print(fail_count)