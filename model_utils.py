from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch
import os
from transformers import AutoModelForCausalLM, AutoConfig

#code adapted from https://github.com/mallorbc/GPTNeoX20B_HuggingFace/blob/main/main.py
#Note: this only works with two devices (48Gb) available and ~72Gb memory
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def init_gpt_neox(fp16=False):
    model_name = "EleutherAI/gpt-neox-20b"
    weights_path = "/gscratch/scrubbed/haoqik/GPT-NeoX"
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
        if fp16:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.save_pretrained(weights_path)
        
    config = AutoConfig.from_pretrained(model_name)

    config.use_cache = False

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    
    device_map = {
		'gpt_neox.embed_in': 0,
		'gpt_neox.layers.0': 0,
		'gpt_neox.layers.1': 0,
		'gpt_neox.layers.2': 0,
		'gpt_neox.layers.3': 0,
		'gpt_neox.layers.4': 0,
		'gpt_neox.layers.5': 0,
		'gpt_neox.layers.6': 0,
		'gpt_neox.layers.7': 0,
		'gpt_neox.layers.8': 0,
		'gpt_neox.layers.9': 0,
		'gpt_neox.layers.10': 0,
		'gpt_neox.layers.11': 0,
		'gpt_neox.layers.12': 0,
		'gpt_neox.layers.13': 0,
		'gpt_neox.layers.14': 0,
		'gpt_neox.layers.15': 0,
		'gpt_neox.layers.16': 0,
		'gpt_neox.layers.17': 0,
		'gpt_neox.layers.18': 0,
		'gpt_neox.layers.19': 0,
		'gpt_neox.layers.20': 0,
		'gpt_neox.layers.21': 0,
		'gpt_neox.layers.22': 1,
		'gpt_neox.layers.23': 1,
		'gpt_neox.layers.24': 1,
		'gpt_neox.layers.25': 1,
		'gpt_neox.layers.26': 1,
		'gpt_neox.layers.27': 1,
		'gpt_neox.layers.28': 1,
		'gpt_neox.layers.29': 1,
		'gpt_neox.layers.30': 1,
		'gpt_neox.layers.31': 1,
		'gpt_neox.layers.32': 1,
		'gpt_neox.layers.33': 1,
		'gpt_neox.layers.34': 1,
		'gpt_neox.layers.35': 1,
		'gpt_neox.layers.36': 1,
		'gpt_neox.layers.37': 1,
		'gpt_neox.layers.38': 1,
		'gpt_neox.layers.39': 1,
		'gpt_neox.layers.40': 1,
		'gpt_neox.layers.41': 1,
		'gpt_neox.layers.42': 1,
		'gpt_neox.layers.43': 1,
		'gpt_neox.final_layer_norm':1,
		'embed_out': 1
	}
    model = load_checkpoint_and_dispatch(
	    model,
	    weights_path,
	    device_map=device_map,
	    offload_folder=None,
	    offload_state_dict=False,
	    dtype="float16"
	)

    return model