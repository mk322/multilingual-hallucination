import json
import os
import argparse
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from urllib.parse import unquote

from model_utils import init_gpt_neox

# Setting up distributed training environment
local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = torch.cuda.device_count()
rank = local_rank
torch.distributed.init_process_group(backend='nccl')

# Define function to print only from the master process
def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)

# Environment settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Setting device based on distributed setup
device = torch.device(f"cuda:{local_rank}")

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str, default='gpt-neo', required=True)
parser.add_argument('-s', '--model_size', type=str, required=True)
#parser.add_argument('--local-rank', type=int, default=0)
parser.add_argument('--cat', type=str, default="Human")

parser.add_argument('--pretrained', type=bool, default=True)

args = parser.parse_args()

pretrain_list = ["bloom", "gpt-neo", "llama"]
finetuned_list = ["vicuna", "alpaca", "chatgpt", "gpt4", "dolly", "bloomz"]

langs = ["en", "es", "zh", "de"]
prompt_langs = {
    "en": "Tell me a bio of {}.",
    "zh": "给我写一关于{}的简介。",
    "es": "Dime una biografía de {}.",
    "de": "Erzähl mir eine Biografie von {}.",
}


category = args.cat

# Open the file for reading
file_name = [f"wiki_data/{f}" for f in os.listdir("wiki_data") if f.startswith(f"{category}_terms")][0]
with open(file_name, 'r', encoding='utf-8') as f:
    # Load the data from the file
    term_dict = json.load(f)
print("load successful")

if args.model_name == "gpt-neo":
    if args.model_size != "20B":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-"+args.model_size)
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-"+args.model_size).to(device)
    else: 
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = init_gpt_neox(True)
elif args.model_name == "bloom":
    if args.model_size != "176b":
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-"+args.model_size)
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-"+args.model_size).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom")
elif args.model_name == "bloomz-mt":
    if args.model_size != "176b":
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-"+args.model_size+"-mt")
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-"+args.model_size+"-mt").to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-mt")
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-mt")

elif args.model_name == "bloomz":
    if args.model_size != "176b":
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-"+args.model_size)
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-"+args.model_size).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz")
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz")
elif args.model_name == "vicuna":
    model = AutoModelForCausalLM.from_pretrained(f"eachadea/vicuna-{args.model_size}-1.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"eachadea/vicuna-{args.model_size}-1.1")
elif args.model_name == "dolly":
    model = AutoModelForCausalLM.from_pretrained(f"databricks/dolly-v2-{args.model_size}").to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"databricks/dolly-v2-{args.model_size}")
else:
  raise Exception("Sorry, the input model name is invalid.")

# Wrapping model with DataParallel
model = torch.nn.DataParallel(model)
model.to(device)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 0

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

#with open(f"result/{category}_generation_{args.model_name}_{args.model_size}.json", "a", buffering=1) as j:
with open(f"result/{category}_generation_{args.model_name}_{args.model_size}.txt", "a", buffering=1) as f:
    print("start")
    for term_link in term_dict:
        for lang in langs:
            term = unquote(term_dict[term_link][lang])
            prompt = prompt_langs[lang].format(term.replace("_", " "))

            inputs = tokenizer(prompt, add_special_tokens=False, padding=True, truncation=True, max_length=128, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256, num_return_sequences=5, return_dict_in_generate=True, top_p=0.9, do_sample=True, output_scores=True)
            #outputs = generator(prompt, max_new_tokens=20, num_return_sequences=5, top_p=0.9, do_sample=True, output_scores=True)
            texts = outputs.sequences
            scores = outputs.scores
            #if not os.path.exists(f'data/scores_{args.model_name}_{args.model_size}'):
                #os.makedirs(f'data/scores_{args.model_name}_{args.model_size}')
            #torch.save(scores, f'data/scores_{args.model_name}_{args.model_size}/{title}_{section}_scores.pt')
            text_list = []
            for i in range(len(texts)):
                text = tokenizer.decode(texts[i], skip_special_tokens=True)[len(prompt):]
                print(f"{term_link}\t{lang}\t{term}\t{i}\t{text}".replace("\n", "\\n"), file=f)
                text_list.append(text)

            #res_pair["generation"] = text_list
            #json.dump(res_pair, j)
            #j.write("\n")
