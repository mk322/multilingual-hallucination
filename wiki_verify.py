import wikipedia
from decompose_facts import decompose

topics = ["Cold_War", "Moon_landing", "Fall_of_the_Western_Roman_Empire", "Fall_of_the_Berlin_Wall", "Industrial_Revolution"]
langs = ["zh", "es"]


for lang in langs:
    wikipedia.set_lang(lang)

    for topic in topics:
        path = f"data/chatgpt/{lang}/{topic}_{lang}.txt"

        with open(path, "r") as f:
            text = f.read()
            with open(f"train_decompose_{lang}.txt", "r") as t:
                prompt = t.read() + text
        texts = generate(prompt, "vicuna", "13b", f"{topic}_{lang}")
        wiki_topic = topic.replace('_', ' ')

        

