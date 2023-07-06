from datasets import load_dataset
import json
dataset = load_dataset("wiki40b", "zh-cn", split="test", beam_runner='DirectRunner')

def find_mid(text, first, last):
    idx1 = text.find(first)
    idx2 = text.find(last)
    
    # length of substring 1 is added to
    # get string from next character
    res = text[idx1 + len(first): idx2]
    return res, text[idx2:]

def process_article(text):
    text = text.replace("\n", "")
    if text.find('_START_SECTION_') < text.find('_START_PARAGRAPH_') and text.find('_START_SECTION_') != -1:
        article_title, rest_string = find_mid(text, '_START_ARTICLE_', '_START_SECTION_')
    else:
        article_title, rest_string = find_mid(text, '_START_ARTICLE_', '_START_PARAGRAPH_')
    #print(article_title)
    processed_sections = []
    while len(rest_string) > 0:
        # section before paragraph
        if rest_string.find('_START_SECTION_') < rest_string.find('_START_PARAGRAPH_') and rest_string.find('_START_SECTION_') != -1:
            section_title, rest_string = find_mid(rest_string, '_START_SECTION_', "_START_PARAGRAPH_")
            #print("ddddddfsf")
            if '_START_SECTION_' in rest_string:
                section_content, rest_string = find_mid(rest_string, '_START_PARAGRAPH_', '_START_SECTION_')
                section_contents = section_content.split("_NEWLINE_")
            else:
                rest_string = rest_string.replace("_START_PARAGRAPH_", "")
                section_contents = rest_string.split("_NEWLINE_")
                rest_string = ""
        else:
            if rest_string.find('_START_SECTION_') == -1:
                section_title = ""
                rest_string = rest_string.replace("_START_PARAGRAPH_", "")
                section_contents = rest_string.split("_NEWLINE_")
                rest_string = ""
            else:
                section_title = ""
                section_content, rest_string = find_mid(rest_string, '_START_PARAGRAPH_', '_START_SECTION_')
                section_contents = section_content.split("_NEWLINE_")
                #rest_string = ""
            
        #print(rest_string)
        #print()
        #print()

        processed_sections.append({
            'section_title': section_title,
            'section_content': section_contents
        })
    return {
        'article_title': article_title,
        'sections': processed_sections
    }



def process_article2(text):
    # Split on 'START_ARTICLE_' to separate the article title
    article_parts = text.split('START_ARTICLE_')

    # The article title should be the part after 'START_ARTICLE_'
    article_title = article_parts[1].split('_START_SECTION_')[0].strip() if len(article_parts) > 1 else ""

    # Everything after the title is considered as sections/paragraphs
    sections_and_paragraphs = article_parts[1][len(article_title):] if len(article_parts) > 1 else ""
    sections_and_paragraphs = sections_and_paragraphs.split('_START_SECTION_')
    
    processed_sections = []

    for sec_or_para in sections_and_paragraphs:
        if '_START_PARAGRAPH_' in sec_or_para:
            section_parts = sec_or_para.split('_START_PARAGRAPH_')

            # If '_START_PARAGRAPH_' was not the first thing in the string, it's a section title
            if section_parts[0].strip():
                section_title = section_parts[0].strip()
            else:  # Else, it's just a paragraph without a section title
                section_title = ""

            section_content = section_parts[1].split('_NEWLINE_') if len(section_parts) > 1 else []
        else:
            section_title = ""
            section_content = sec_or_para.split('_NEWLINE_')

        # Stripping leading and trailing white spaces from each paragraph
        section_content = [paragraph.strip() for paragraph in section_content]

        processed_sections.append({
            'section_title': section_title,
            'section_content': section_content
        })

    return {
        'article_title': article_title,
        'sections': processed_sections
    }



processed_articles = []
for i in range(len(dataset)):
#for i in range(50):

    text = dataset[i]["text"]
    art_dict = process_article(text)

    processed_articles.append(art_dict)

# Write the processed articles to a JSON file
with open('data/zh-cn_articles.json', 'w', encoding='utf-8') as f:
    json.dump(processed_articles, f, ensure_ascii=False, indent=4)