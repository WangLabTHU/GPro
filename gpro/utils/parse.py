import os
import shutil
import signal
import sys

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# from ruamel.yaml import YAML

import torch
import torch.cuda
from pathlib import Path

# parse parameter set <key> for model <model>, with optional additional parameters
class Bunch:

    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Bunch(v))
            else:
                setattr(self, k, v)


# update base dict <d> with values in dict <u>
def update_base(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_base(d.get(k, {}), v)
        elif k not in d.keys():
            d[k] = v
    return d


# recursively resolve 'base' keys in parameters
def resolve_params(allparams, param_id):
    params = {}
    update_base(params, allparams[param_id])
    while params['base'] is not None:
        update_base(params, allparams[params['base']])
        params['base'] = allparams[params['base']]['base']
    return params


# parse parameter set <key> for model <model>, with optional additional parameters
def parse_args(model, param_id, project_path, additional=None):
    yaml = YAML(typ='safe')
    with open(os.path.join(project_path, 'tasks/args.yaml')) as f:
        allparams = yaml.load(f)[model]
    params = resolve_params(allparams, param_id)
    if additional is not None:
        params.update(vars(additional))
    return Bunch(params)


# called by each main script at the start
def startup_predictor(model, type, project_path, additional=None):
    signal.signal(signal.SIGINT, lambda *_: sys.exit(1))
    p = parse_args(model, type, project_path, additional)
    if 'submodels' in p.__dict__:
        for m, mp in p.submodels.__dict__.items():
            setattr(p, m, parse_args(m, mp.id))
    # set random seeds
    torch.manual_seed(p.seed)
    np.random.seed(p.seed)
    return p

def get_project_root():
    return Path(__file__).parent.parent