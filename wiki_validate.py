import wikipedia
from decompose_facts import generate
from rank_bm25 import BM25Okapi
key_word_dic = {
    "Cold_War": 
        {
        "zh":"冷战",
        "es": "guerra Fría"
        },
    "Moon_landing":
        {
        "zh":"登月",
        "es": "Alunizaje"
        },
    "Fall_of_the_Western_Roman_Empire":
        {
        "zh":"罗马帝国的衰落",
        "es": "Caída del Imperio romano de Occidente"
        }, 
    "Fall_of_the_Berlin_Wall":
        {
        "zh":"柏林墙倒塌",
        "es": "Caída del Muro de Berlín"
        }, 
    "Industrial_Revolution":{
        "zh":"第一次工业革命",
        "es": "Revolución Industrial"
        }
    }

key_word_dic = {
    "Cold_War": 
        {
        "zh":"冷战",
        "es": "guerra Fría"
        }
}

langs = ["zh", "es"]
langs = ["es"]

for lang in langs:
    wikipedia.set_lang(lang)
    for topic in key_word_dic:
        path = f"data/chatgpt/{lang}/{topic}_{lang}_atomic.txt"
        key_word = key_word_dic[topic][lang]
        print(key_word)
        content = wikipedia.page(key_word).content
        #with open(f"data/chatgpt/{lang}/{topic}_{lang}_wiki.txt", "w") as f:
            #print(content, file=f)
        if lang=="zh":
            sent_content = content.split("。")
        elif lang == "es":
            sent_content = content.replace("\n", "").split(".")
        print(len(sent_content))
        tokenized_corpus = [doc.split(" ") for doc in sent_content]

        bm25 = BM25Okapi(tokenized_corpus)

        with open(path, "r") as a:
            facts = a.read().splitlines()
        with open(f"train_verify_{lang}.txt", "r") as f:
            demo = f.read()
        for i in range(len(facts)):
            fact = facts[i]
            # bug
            if lang=="zh":
                tokenized_query = fact.split("")
            elif lang == "es":
                tokenized_query = fact.split(" ")
            evidence = " ".join(bm25.get_top_n(tokenized_query, sent_content, n=20))
            #print(evidence)

            the_demo = demo.replace("<context_to_be_checked>", evidence)
            the_demo = demo.replace("<statement_to_be_checked>", fact)
            with open("demo_es.txt", "a") as o:
                print(the_demo, file=o)
                print("<END_OF_EX>", file=o)
            prompt = the_demo + ""
            texts = generate(prompt, "vicuna", "13b", f"{topic}_{lang}_verify{i}")

