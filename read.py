import json

path = "multilingual-hallucination/data/zh-cn_articles_generation.json"

with open(path, "r") as f:
    dic = json.load(path)

print(dic)