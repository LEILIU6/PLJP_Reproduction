#!/usr/bin/env python3
# -*- encoding: utf-8 -*-


import json
import codecs
from sklearn.metrics import precision_score, recall_score

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

testset_path = "data/testset/cjo22/testset.json"
k = 3

data = []
with open(testset_path, encoding="utf8") as f:
    for line in f.readlines():
        obj = json.loads(line)
        data.append(obj)

y_true = []
for case in data:
    ar = pt_cls2str[get_pt_cls(int(case["meta"]["penalty"]))]
    y_true.append(ar)

topk_predictive_result_path = "data/output/domain_model_out/candidate_label/cjo22/BERTC/penalty_topk_6w.json"
topk = 3
y_pred_topk = []
y_pred = []
with codecs.open(topk_predictive_result_path, "r", "utf-8") as f:
    data = json.load(f)
    for d_i in data:
        d_i = [x for x in d_i if x != "其他"]
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
