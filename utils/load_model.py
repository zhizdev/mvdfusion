'''
Functions to load models
'''
from collections import OrderedDict
import importlib
from omegaconf import OmegaConf
import torch


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    ignore_set = {'__is_first_stage__', '__is_unconditional__', 'dataset', 'trainer', 'saver'}
    if not "target" in config:
        if config in ignore_set:
            return None
        print('*EEROR', config)
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, ckpt=None, verbose=False, replace_key=None, ignore_keys=[], param_mapper={}, remove_keys=[]):

    model = instantiate_from_config(config)

    if ckpt is not None:

        #@ LOAD CHECKPOINT
        print(f"Loading {config['target']} from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            if verbose:
                print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]

        #@ RENAME STATE DICT PARAMS
        sd_ = OrderedDict()

        if verbose:
            parent_set = set()
            for k, v in sd.items():
                k_ = k[:k.find('.')]
                parent_set.add(k_)        
            print('ckpt contains:', parent_set)

        for k, v in sd.items():
            if replace_key is not None:
                name = k.replace(replace_key[0], replace_key[1])
            else:
                name = k

            if name in param_mapper:
                if verbose:
                    print(f'mapping {name} to {param_mapper[name]}')
                name = param_mapper[name]

            if any(name == remove_k for remove_k in remove_keys):
                print('REMOVING WEIGHT', name)
                continue 

            sd_[name] = v
    
        m, u = model.load_state_dict(sd_, strict=False)

        #@ PRINT MISSING KEYS
        missing_core = False
        if len(m) > 0:
            parent_set = set()
            for uk in m:
                uk_ = uk[:uk.find('.')]
                if any(ignore_key in uk for ignore_key in ignore_keys):
                    pass
                else:
                    print('missing core:', uk)
                    missing_core = True
                if uk_ not in parent_set:
                    parent_set.add(uk_)
            if verbose:
                print('missing root keys:', parent_set)
        else:
            if verbose:
                print('all weights found')

        # assert not missing_core, 'Missing core parameters while loading {config["target"]} from {ckpt}'
        if missing_core:
            print(f'***\n***CRITICAL WARNING\nMissing core parameters while loading {config["target"]} from {ckpt}')

        #@ SKIPPING UNEXPECTED KEYS
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(len(u))
            parent_set = set()
            for uk in u:
                uk_ = uk[:uk.find('.')]
                print('unexpected:', uk)
                if uk_ not in parent_set:
                    parent_set.add(uk_)
            print(parent_set)

    else:
        if verbose:
            print('initializing from scratch')

    model.eval()
    return model