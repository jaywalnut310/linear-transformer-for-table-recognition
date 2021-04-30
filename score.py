import os
import json
import tqdm
import glob
import numpy as np
from srcs.PubTabNet.src.metric import TEDS
from srcs.PubTabNet.src.parallel import parallel_process


f_list = glob.glob("./results/val1/*.json")
pred = {}
for f_name in f_list:
    with open(f_name, "r") as f:
        pred.update(json.load(f))

with open("/data/private/datasets/pubtabnet/annotations/val.json", "r") as f:
    data = json.load(f)
true = [(x['image_path'].split("/")[-1], "".join(x['text'][1:-1])) for x in data if x['image_path'].split("/")[-1] in pred]

true_sorted = sorted(true, key=lambda x: len(x[1]))

teds = TEDS(n_jobs=48)

html_strings_pred = [pred[x[0]] for x in true_sorted]
prefix = '<html><body><table>'
postfix = '</table></body></html>'
html_strings_tgt = [prefix + x[1] + postfix for x in true_sorted]

inputs = [{"pred": pred, "true": true} for pred, true in zip(html_strings_pred, html_strings_tgt)]
scores = parallel_process(inputs, teds.evaluate, use_kwargs=True, n_jobs=teds.n_jobs, front_num=1)
print(np.mean(scores))
