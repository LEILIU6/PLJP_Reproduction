#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


import json
import codecs
from sklearn.metrics import precision_score, recall_score

testset_path = "data/testset/cjo22/testset.json"
k = 3

data = []
with open(testset_path, encoding="utf8") as f:
    for line in f.readlines():
        obj = json.loads(line)
        data.append(obj)

y_true = []
for case in data:
    ar = case["meta"]["accusation"][0]
    y_true.append(ar)

topk_predictive_result_path = "data/output/domain_model_out/candidate_label/cjo22/BERTC/charge_topk_6w.json"
topk = 3
y_pred_topk = []
y_pred = []
with codecs.open(topk_predictive_result_path, "r", "utf-8") as f:
    data = json.load(f)
    for d_i in data:
        y_pred_topk.append(d_i[:k])
        y_pred.append(d_i[0])

correct = 0
total = 0
for i, true_i in enumerate(y_true):
    total += 1
    pred_i = y_pred_topk[i]
    if true_i in pred_i:
        correct += 1
print(f"top-{k} accuracy: {correct/total}")
print(f"macro precision: {precision_score(y_true, y_pred, average='macro')}")
print(f"macro recall: {recall_score(y_true, y_pred, average='macro')}")
