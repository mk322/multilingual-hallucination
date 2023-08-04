
#metric_list = ["rouge1", "rouge2", "rougeL", "rougeLsum", "Unigram"]
metric_list = ["Unigram", "BLEU"]

for metric in metric_list:
    path = f"result/scores/{metric}_Human_bloomz-mt_non-en.txt"

    with open(path, "r") as f:
        txt = f.read().splitlines()

    lang_dict = {}

    for line in txt[1:]:
        parts = line.split("\t")
        lang = parts[1]
        score = parts[3]
        if lang not in lang_dict:
            lang_dict[lang] = []

        lang_dict[lang].append(float(score))

    with open(f"result/scores/overview/overview_{metric}_Human_bloomz-mt_non-en.txt", "w") as f:
        for lang in lang_dict:
            st_mean = lang
            st_sum = lang
            #for i in range(12):
            s = round(sum(lang_dict[lang]), 3)
            average = round(sum(lang_dict[lang]) / len(lang_dict[lang]), 6)
            st_mean += " " + str(average)
            st_sum += " " + str(s)
            print(st_mean, file=f)
            print(st_sum, file=f)

