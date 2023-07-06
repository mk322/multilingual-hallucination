import json
import os
import matplotlib.pyplot as plt
from urllib.parse import unquote

page_dict = {}
page_dict_min = {}

files = [f"wiki_data/{f}" for f in os.listdir("wiki_data") if f.endswith("by_page_views.json")]
for file in files:
    cat_name = file.split("/")[1].split("_")[0]
    #page_dict[cat_name] = {}
    page_dict[cat_name] = {}
    lowest_sum_term_link = None
    lowest_sum = 0

    with open(file, "r", encoding="utf-8") as f:
        category_data = json.load(f)

        for lang in category_data:
            #page_dict_min[cat_name][lang] = float('inf')  # Initialize with infinity for comparison
           # term_link_min = ''

            for term in category_data[lang]:
                term_link = term[0]
                pageview = term[1]
                name = term[2]
                if not term_link in page_dict[cat_name]:
                    page_dict[cat_name][term_link] = {}
                page_dict[cat_name][term_link][lang] = pageview


    for term_link in page_dict[cat_name]:
        pageviews_sum = sum(page_dict[cat_name][term_link].values())

        if pageviews_sum > lowest_sum:
            lowest_sum = pageviews_sum
            lowest_sum_term_link = term_link

    #print("Term Link with Lowest Sum of Pageviews:", lowest_sum_term_link)
    #print("Sum of Pageviews:", lowest_sum)

    languages = list(page_dict[cat_name][term_link].keys())
    pageviews = [page_dict[cat_name][lowest_sum_term_link][l] for l in languages]
    #print(languages)
"""
    plt.figure(figsize=(10, 6))
    plt.bar(languages, pageviews)
    term_name = unquote([ele[2] for ele in category_data["en"] if ele[0]==lowest_sum_term_link][0].replace("_", " "))
    print(term_name)
    plt.title(f'Pageviews Distribution for the most frequent term {term_name} for {cat_name}')
    plt.xlabel('Language')
    plt.ylabel('Pageviews')
    plt.xticks(rotation=45)
    #plt.show()
    plt.savefig(f"wiki_data/figures/{cat_name}_max.png")


# Create a figure for each category
for category, values in page_dict_min.items():
    languages = list(values.keys())
    pageviews = list(values.values())

    plt.figure(figsize=(10, 6))
    plt.bar(languages, pageviews)
    plt.title(f'Maximum Pageviews for {category}')
    plt.xlabel('Language')
    plt.ylabel('Maximum Pageviews')
    plt.xticks(rotation=45)
    plt.savefig(f"wiki_data/figures/{category}_max.png")
"""
categories = ['Human', 'City', 'Film', 'Country', 'Language']
examples = [576, 537, 221, 187, 124]

plt.bar(categories, examples)
plt.xlabel('Category')
plt.ylabel('Number of Examples')
plt.title('Number of Examples per Category')

# Add labels to each bar
for i, v in enumerate(examples):
    plt.text(i, v, str(v), ha='center', va='bottom')

plt.savefig(f"wiki_data/figures/counts.png")

