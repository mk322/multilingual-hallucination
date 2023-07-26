import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

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

key_list = [
    'rouge1_fmeasure', 'rouge1_precision', 'rouge1_recall', 
    'rouge2_fmeasure', 'rouge2_precision', 'rouge2_recall', 
    'rougeL_fmeasure', 'rougeL_precision', 'rougeL_recall', 
    'rougeLsum_fmeasure', 'rougeLsum_precision', 'rougeLsum_recall'
    ]

def get_second_element_by_key(data, key):
    '''
    Function to get the second element in a sublist for a certain key.
    
    Args:
        data (list): List of sublists.
        key (str): The key to look for in the first element of each sublist.
        
    Returns:
        The second element in the sublist if the key is found, or None if the key is not found.
    '''

    for sublist in data:
        if sublist[0] == key:
            return sublist[1]

    return None


def plot_metric_vs_page_views(page_views, metric_values, languages, target_languages, name):
    '''
    Function to plot the correlation between page views and a given metric, colored by language.
    
    Args:
        page_views (list): List of page views.
        metric_values (list): List of metric values.
        languages (list): List of languages.
        target_languages (list): List of languages to be included in the plot.
        
    Returns:
        A scatter plot.
    '''

    # Initialize plot
    plt.figure(figsize=(10, 6))

    # For each target language, plot the points corresponding to that language
    for lang in target_languages:
        lang_indices = [i for i, x in enumerate(languages) if x == lang]
        lang_page_views = [float(page_views[i]) for i in lang_indices]
        lang_metric_values = [float(metric_values[i]) for i in lang_indices]

        plt.scatter(lang_page_views, lang_metric_values, label=lang)

    # Add title and labels
    plt.title('Page Views vs. Metric by Language')
    plt.xlabel('Page Views')
    plt.ylabel('Metric')
    plt.legend()

    # Show the plot
    plt.savefig(f"result/scores/fig/{name}.png")


kk = 0
df = {}
with open("wiki_data/Human_by_page_views.json", "r") as f:
    page_view = json.load(f)

path = "result/scores/ROUGE_Human_bloomz-mt_non-en.txt"

with open(path, "r") as f:
    txt = f.read().splitlines()

lang_dict = {}

link_list = set()
for line in txt[1:]:
    parts = line.split("\t")
    lang = parts[1]
    link = parts[0]
    scores = [i for i in parts[3:]]
    score = parts[3]
    link_list.add(link)
    #score = parts[3]
    if lang not in lang_dict:
        lang_dict[lang] = {}
    if link not in lang_dict[lang]:
        lang_dict[lang][link] = [[] for _ in range(12)]
    for i in range(12):
        score = scores[i]

        lang_dict[lang][link][i].append(float(score))

for kk in range(12):
    pageviews = []
    metrics = []
    langs = []
    for link in link_list:
        for lang in lang_names:
            pageviews.append(get_second_element_by_key(page_view[lang], link))
            metrics.append(max(lang_dict[lang][link][kk]))
            langs.append(lang)

    #print(pageviews)
    #print(metrics)
    print(len(pageviews), len(metrics), len(langs))

    plot_metric_vs_page_views(pageviews, metrics, langs, ["zh", "ro", "en", "ru", "es", "sv"], key_list[kk])