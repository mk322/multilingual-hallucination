import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import argparse
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def generate(prompt, model_name, model_size, label=""):
    if model_name == "vicuna":
        model = AutoModelForCausalLM.from_pretrained(f"eachadea/vicuna-{model_size}-1.1", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(f"eachadea/vicuna-{model_size}-1.1")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = 0

        #generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        inputs = tokenizer(prompt, add_special_tokens=False, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        num_prompt_tokens = len(input_ids[0])

        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=1024, num_return_sequences=5, return_dict_in_generate=True, do_sample=True, output_scores=True)
        #outputs = generator(prompt, max_new_tokens=20, num_return_sequences=5, top_p=0.9, do_sample=True, output_scores=True)
        texts = outputs.sequences
        scores = outputs.scores

        if not os.path.exists(f'data/early_scores_{model_name}_{model_size}'):
            os.makedirs(f'data/early_scores_{model_name}_{model_size}')
        if not os.path.exists(f'data/atomic_facts'):
            os.makedirs(f'data/atomic_facts')
        torch.save(scores, f'data/early_scores_{model_name}_{model_size}/scores_{label}.pt')
        text_list = []
        with open(f"data/es/atomic_facts/{label}.txt", "a") as f:
            for i in range(len(texts)):
                text = tokenizer.decode(texts[i][num_prompt_tokens:], skip_special_tokens=True)
                print(f"{i}_{text}".replace("\n", "\\n"), file=f)
                text_list.append(text)
    return text_list
