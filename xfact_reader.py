import pandas as pd
from googletrans import Translator

path = "data/x-fact-including-en/test.all.tsv"
df = pd.read_csv(path, sep='\t', header=0, quotechar='"', on_bad_lines='skip')
translator = Translator()
translation = translator.translate('veritas lux mea')
print(translation)
#print(df.head())
#print(len(df))

examples = []
with open(path, 'r') as fp:
    next(fp)
    for line in fp:
        arr = line.strip().split('\t')
        lang = arr[0].lower()
        site = arr[1].lower()
        claim = arr[-2][1:-1]
        label = arr[-1].lower()
        claimant = arr[-3]
        evidences = [s for s in arr[2:7] if s != "<DUMMY_EVIDENCE>"]
        dic = {
            "lang": lang,
            "claim": claim,
            "label": label,
            "evidences": evidences,
            "site": site,
            "claimant" : claimant
        }
        #if lang == "es":
            #print("claim:", claim)
            #print("\n".join(evidences))
            #print(label)
        #print(claim)

        #examples.append(dic)
    

#print(examples[:3])

print(len(examples))