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
resp_file = "data/output/llm_out/cjo22/BERTC/article/3shot/gpt-4o-mini_6w.json"

with open(resp_file, encoding="utf8") as f:
    for i, line in enumerate(f.readlines()):
        obj = json.loads(line)
        results.append(obj)

y_true = []

for case in data:

    # # author original code
    # ar = int(max(case["meta"]["relevant_articles"]))
    # # endof author original code

    ar = int(case["meta"]["relevant_articles"][0])
    y_true.append(ar)


y_pred = []
count = 0
for obj in results:
    count += 1
    # resp = obj["choices"][0]["text"] # dav
    resp = obj["llm_response"]["choices"][0]["message"]['content'] # turbo
    res = re.findall("第(.*?)条", resp)
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

    # # author original code
    # y_pred.append(max(ars))
    # # endof author original code
    
    y_pred.append(ars[0])

acc, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
map, mar, maf, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

acc = round(acc, 6) * 100
map = round(map, 6) * 100
mar = round(mar, 6) * 100
maf = round(maf, 6) * 100

print(f"acc:{acc}, map:{map}, mar:{mar}, maf:{maf}")


print(len(y_pred))
print(y_pred[:20])
print(len(y_true))
print(y_true[:20])