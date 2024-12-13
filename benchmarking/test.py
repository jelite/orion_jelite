import json
import pandas as pd

with open(f"/workspace/exps/be_infer/traces-rps85.05-reqs5000-num{0}.json", "r") as f:
    arrival_times = pd.read_json(f)

import pdb; pdb.set_trace()