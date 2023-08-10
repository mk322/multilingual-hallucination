import argparse
import gc
import math
import os
import time
import json
import torch
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from urllib.parse import unquote


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
    parser.add_argument("--name", type=str, help="Name path", required=True)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, help="float16 or int8", choices=["int8", "float16"], default="int8")
    parser.add_argument('--en_prompt', type=bool, default=False)
    parser.add_argument('--cat', type=str, default="Human")
    return parser.parse_args()


t_start = time.time()

num_tokens = 100

args = get_args()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = torch.cuda.device_count()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

rank = local_rank


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


lang_names = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "de": "German",
    "ru": "Russian",
    "id": "Indonesian",
    "vi": "Vietnamese",
    "fa": "Persian",
    "uk": "Ukrainian",
    "sv": "Swedish",
    "th": "Thai",
    "ja": "Japanese",
    "ro": "Romanian",
    "hu": "Hungarian",
    "bg": "Bulgarian",
    "fr": "French",
    "fi": "Finnish",
    "ko": "Korean",
    "it": "Italian",
}
prompt_langs = {
    "en": "Tell me a bio of {}.",
    "zh": "给我写一关于{}的简介。",
    "es": "Dime una biografía de {}.",
    "de": "Erzähl mir eine Biografie von {}.",
    "ru": "Расскажите мне биографию {}.",
    "id": "Beri tahu saya biografi {}.",
    "vi": "Hãy cho tôi biết tiểu sử của {}.",
    "fa": "بیوگرافی {} را به من بگویید.",
    "uk": "Розкажіть мені біографію {}.",
    "sv": "Berätta en biografi om {}.",
    "th": "บอกเล่าประวัติของ {}",
    "ja": "{} の略歴を教えてください。",
    "ro": "Spune-mi o biografie a lui {}.",
    "hu": "Mondja el {} életrajzát.",
    "bg": "Разкажи ми биография на {}.",
    "fr": "Dites-moi une biographie de {}.",
    "fi": "Kerro minulle henkilön {} elämäkerta.",
    "ko": "{}의 약력을 알려주세요.",
    "it": "Raccontami una biografia di {}.",
}

category = args.cat

# Open the file for reading
file_name = [f"wiki_data/{f}" for f in os.listdir("wiki_data") if f.startswith(f"{category}_terms")][0]
with open(file_name, 'r', encoding='utf-8') as f:
    # Load the data from the file
    term_dict = json.load(f)
print("load successful")


print_rank0(f"Using {world_size} gpus")
model_name = args.name
print_rank0(f"Loading model {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 0

dtype = torch.bfloat16 if model_name in ["bigscience/bloom", "bigscience/bloomz-mt", "bigscience/bigscience-small-testing"] else torch.float16

# print(get_max_memory_per_gpu_dict())

infer_dtype = args.dtype
if infer_dtype == "int8":
    dtype = torch.int8

kwargs = dict(
    device_map="auto",
)


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


# balanced_low_0 - because it allows a larger batch size with multiple GPUs
if get_world_size() > 1:
    kwargs["device_map"] = "balanced_low_0"


if infer_dtype == "int8":
    print_rank0("Using `load_in_8bit=True` to use quanitized model")
    kwargs["load_in_8bit"] = True
else:
    kwargs["torch_dtype"] = dtype


model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


### Generate
def generate(inputs):
    """returns a list of zipped inputs, outputs and number of new tokens"""

    generate_kwargs = dict(max_new_tokens=256, num_return_sequences=5, return_dict_in_generate=True, top_p=0.9, do_sample=True, output_scores=True)

    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda:0")

    outputs = model.generate(**input_tokens, **generate_kwargs)
    scores = outputs.scores
    seq = outputs.sequences

    # Get lengths of original inputs
    input_lengths = [len(input_token) for input_token in input_tokens["input_ids"]]

    # Repeat each input length num_return_sequences times
    input_lengths = [input_len for input_len in input_lengths for _ in range(generate_kwargs['num_return_sequences'])]

    # Initialize list to store the generated texts after the prompt
    texts = []

    # For each generated sequence, remove the tokens corresponding to the original input
    for idx, sequence in enumerate(seq):
        input_len = input_lengths[idx]
        generated_sequence = sequence[input_len:]  # Only consider tokens after the original input
        generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        texts.append(generated_text)

    return texts


if args.en_prompt:
    suffix = "en"
else:
    suffix = "non-en"

print("start")
prompt_dict = {}
prompt_list = []
for term_link in term_dict:
    for lang in list(prompt_langs.keys()):

        term = unquote(term_dict[term_link][lang])
        text_term = term.replace("_", " ")
        if not args.en_prompt:
            prompt = prompt_langs[lang].format(text_term) + f"\n\n{text_term}"
        else:
            prompt = f"Tell me a biography of {text_term} in {lang_names[lang]}."
        prompt_dict[prompt] = (term_link, lang, term) 
        prompt_list.append(prompt)

with open(f"result/{category}_generation_bloom_{suffix}.txt", "a", buffering=1) as f:
    for i in range(10327, len(prompt_list), args.batch_size):
        inputs = prompt_list[i : i+args.batch_size]
    #if not os.path.exists(f'data/scores_{args.model_name}_{args.model_size}'):
        #os.makedirs(f'data/scores_{args.model_name}_{args.model_size}')
    #torch.save(scores, f'data/scores_{args.model_name}_{args.model_size}/{title}_{section}_scores.pt')
        texts = generate(inputs)
        for j in range(args.batch_size):
            for i in range(5):
                text = texts[j * 5 + i]
                term_link, lang, term = prompt_dict[inputs[j]]

                print(f"{term_link}\t{lang}\t{term}\t{i}\t{text}".replace("\n", "///n"), file=f)