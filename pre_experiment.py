import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import argparse
import torch
from urllib.parse import unquote

from model_utils import init_gpt_neox
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str, default='gpt-neo', required=True)
parser.add_argument('-s', '--model_size', type=str, required=True)
parser.add_argument('--en_prompt', type=bool, default=False)
parser.add_argument('--cat', type=str, default="Human")

args = parser.parse_args()

pretrain_list = ["bloom", "gpt-neo", "llama"]
finetuned_list = ["vicuna", "alpaca", "chatgpt", "gpt4", "dolly", "bloomz"]

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

if args.model_name == "gpt-neo":
    if args.model_size != "20B":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-"+args.model_size)
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-"+args.model_size).to(device)
    else: 
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = init_gpt_neox(True)
elif args.model_name == "bloom":
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-"+args.model_size)
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-"+args.model_size).to(device)
elif args.model_name == "bloomz-mt":
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-"+args.model_size+"-mt")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-"+args.model_size+"-mt").to(device)
elif args.model_name == "bloomz":
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-"+args.model_size)
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-"+args.model_size).to(device)
elif args.model_name == "vicuna":
    model = AutoModelForCausalLM.from_pretrained(f"eachadea/vicuna-{args.model_size}-1.1").to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"eachadea/vicuna-{args.model_size}-1.1")
elif args.model_name == "dolly":
    model = AutoModelForCausalLM.from_pretrained(f"databricks/dolly-v2-{args.model_size}").to(device)
    tokenizer = AutoTokenizer.from_pretrained(f"databricks/dolly-v2-{args.model_size}")
else:
  raise Exception("Sorry, the input model name is invalid.")


tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 0

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

if args.en_prompt:
    suffix = "en"
else:
    suffix = "non-en"

#with open(f"result/{category}_generation_{args.model_name}_{args.model_size}.json", "a", buffering=1) as j:
with open(f"result/{category}_generation_{args.model_name}_{args.model_size}_{suffix}.txt", "a", buffering=1) as f:
    print("start")
    for term_link in term_dict:
        for lang in list(prompt_langs.keys()):
            term = unquote(term_dict[term_link][lang])
            if not args.en_prompt:
                prompt = prompt_langs[lang].format(term)
            else:
                prompt = f"Tell me a biography of {term} in {lang_names[lang]}."

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
