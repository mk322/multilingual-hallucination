import py3langid as langid
from langdetect import detect_langs
from langdetect import DetectorFactory
DetectorFactory.seed = 0

def py3lang_detect(text):
    # identified language and probability
    lang, prob = langid.classify(text)
    # all potential languages
    ranks = langid.rank(text)

    return (lang, prob, ranks)

def langdects(text):

    ranks = detect_langs(text)
    #print(type(ranks[0]))
    lang = ranks[0].lang
    prob = ranks[0].prob
    return (lang, prob, ranks)

path = "result/Human_generation_bloomz-mt_non-en.txt"

with open(path, "r") as f:
    txt = f.read().splitlines()

lang_dict = {}
correct = 0
top5_correct = 0
top2_correct = 0
top4_correct = 0

top3_correct = 0
valid = 0
link_list = set()
with open("result/langdetect/py3lang_Human_bloomz-mt_non-en.txt", "w") as f:
    for i in range(len(txt)):
        eles = txt[i].split("\t")
        link = eles[0]
        entity = eles[2]
        time = eles[3]
        text = "\t".join(eles[4:]).replace("///n", "\n")
        gold_lang = eles[1]
        if text:
            valid += 1
            ret = py3lang_detect(text)
            pred_lang = ret[0]
            #ret = langdects(text)
            ret_string = ""
            for ele in ret[2]:
                ret_string += ele[0] + "__" + str(ele[1]) + "\t"
                #ret[]+"___"+str(round(ret[1],6))
            if pred_lang == gold_lang:
                correct += 1

            top_langs = [lang_t[0] for lang_t in ret[2]]
            if gold_lang in top_langs[:2]:
                top2_correct += 1
            if gold_lang in top_langs[:3]:
                top3_correct += 1
            if gold_lang in top_langs[:4]:
                top4_correct += 1
            if gold_lang in top_langs[:5]:
                top5_correct += 1
        else:
            ret_string = "NaN"
        print(f"{link}\t{gold_lang}\t{entity}\t{time}\t{ret_string}", file=f)

print(correct, top2_correct, top3_correct, top4_correct, top5_correct)
print(valid)
print(correct / valid)