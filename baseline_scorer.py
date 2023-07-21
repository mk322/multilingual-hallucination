from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.functional.text import bleu_score
import nltk
#from tokenizer import split_into_sentences
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
import pycountry

key_list = [
    'rouge1_fmeasure', 'rouge1_precision', 'rouge1_recall', 
    'rouge2_fmeasure', 'rouge2_precision', 'rouge2_recall', 
    'rougeL_fmeasure', 'rougeL_precision', 'rougeL_recall', 
    'rougeLsum_fmeasure', 'rougeLsum_precision', 'rougeLsum_recall'
    ]

def get_language_name(lang_code):
    try:
        lang = pycountry.languages.get(alpha_2=lang_code)
        return lang.name.lower()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def bleu_scorer(pred, target):

    return bleu_score(pred, [target]).item()

def rouge_score(preds, target):
    rouge = ROUGEScore(use_stemmer=True)

    dic = rouge(preds, target)

    ret_list = []
    ret_str = ""
    for key in key_list:
        num = dic[key].item()
        ret_list.append(num)
        ret_str += str(num) + "\t"
    return ret_str[:-1]

# Function to compute unigram overlap
def unigram_overlap(reference_text, source_text, lang):
    # Get English stop words
    stop_words = set(stopwords.words(lang if lang in stopwords.fileids() else 'english'))

    # Tokenize the texts into words
    reference_words = set(nltk.word_tokenize(reference_text))
    source_words = set(nltk.word_tokenize(source_text))
    
    # Remove stop words
    reference_words = reference_words - stop_words
    source_words = source_words - stop_words
    
    # Compute the unigram overlap
    overlap = len(reference_words & source_words) / len(reference_words | source_words)

    return overlap


refer_path = "result/wiki_check/Human_wiki_summary.txt"
source_path = "result/Human_generation_bloomz-mt_non-en.txt"

with open(source_path,"r", encoding="utf-8",) as f:
    list_lines = f.read().splitlines()

with open(refer_path,"r", encoding="utf-8",) as f:
    refer_lines = f.read().splitlines()
    refer_dic = {}
    for line in refer_lines:
        parts = line.split("\t")
        link = parts[0]
        lang = parts[1]
        text = parts[2]
        refer_dic[(link, lang)] = text

with open("result/scores/Unigram_Human_bloomz-mt_non-en.txt", "w", buffering=1) as f:
    for i in range(0, len(list_lines), 5):
        link, lang, term, num, text = list_lines[i].split("\t")
        sentences = [list_lines[i+j].split("\t")[-1] for j in range(5)]
        refer_text = refer_dic[(link, lang)]
        for k in range(5):
            source_sent = sentences[k]
            #ret_str = bleu_score(source_sent, split_into_sentences(refer_text.replace("///n", "\n")))
            ret_str = unigram_overlap(source_sent, refer_text.replace("///n", "\n"), get_language_name(lang))
            #result_str = "\t".join(result_dic[key] for key in key_list)
            print(f"{link}\t{lang}\t{k}\t{ret_str}", file=f)
            #text = max(sentences, key=len)
