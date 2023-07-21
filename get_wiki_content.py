import wikipedia


file_name = "result/Human_generation_bloomz-mt_non-en.txt"

with open(file_name,"r", encoding="utf-8",) as f:
    list_lines = f.read().splitlines()

with open("result/wiki_check/Human_wiki_summary.txt", "a", buffering=1) as fo:
    for i in range(0, len(list_lines), 5):
        link, lang, term, num, text = list_lines[i].split("\t")
        sentences = [list_lines[i+j].split("\t")[-1] for j in range(5)]
        text = max(sentences, key=len)
        wikipedia.set_lang(lang)
        term = term.replace("_", " ")
        #print(term)
        #summary = wikipedia.summary(term, auto_suggest=False, sentences=10)
        summary = wikipedia.page(title=term, auto_suggest=False).summary.replace("\n", "///n")
        print(f"{link}\t{lang}\t{term}\t{summary}", file=fo)

