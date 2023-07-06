import requests
import json
from collections import defaultdict

full_categories = {
    "human": "Q5",
    "City": "Q515",
    "Country": "Q6256",
    "River": "Q4022",
    "Mountain": "Q8502",
    "Company": "Q783794",
    "University": "Q3918",
    "Non-profit organization": "Q163740",
    "Government agency": "Q327333",
    "War": "Q198",
    "Election": "Q40231",
    "Sports event": "Q1656682",
    "Concert": "Q59861",
    "Book": "Q571",
    "Film": "Q11424",
    "Music album": "Q482994",
    "Video game": "Q7889",
    "Language": "Q34770",
    "Religion": "Q9174",
    "Scientific concept": "Q151885",
    "Economic concept": "Q8134",
    "Building": "Q41176",
    "Artwork": "Q838948",
    "Vehicle": "Q42889",
    "Tool": "Q39546",
    "Plant": "Q756",
    "Animal": "Q729",
    "Bacteria": "Q10876",
    "Fungus": "Q25433"
}

categories = {
    #"University": "Q3918",
    #"Sports event": "Q1656682",
    #Book": "Q571",
    #"Animal": "Q729",
    #"Plant": "Q756",
    #"Video game": "Q7889",
    #"Artwork": "Q838948",
    #"Vehicle": "Q42889",
    #"Tool": "Q39546",
    #"Language": "Q34770",
    #"Religion": "Q9174",
    #"Scientific concept": "Q151885",
    #"Building": "Q41176",
    #"Event": "Q1656682",
    #"Human": "Q5",
    #"Country": "Q6256",
    #"River": "Q4022",
    #"Mountain": "Q8502",
    #"War": "Q198",
    #"Art": "Q735",
    "Language": "Q34770",
    #"Building": "Q41176",
    #"Organizations": "Q43229"
}

# List of top 20 languages
languages = ['en', 'ru', 'id', 'vi', 'fa', 'uk', 'sv', 'th', 'ja', 'de', 'ro', 'hu', 'bg', 'fr', 'fi', 'ko', 'es', 'it', 'zh']
#languages = ['ko', 'es', 'it', 'pt', 'el']
# Wikidata API endpoint
wikidata_url = 'https://query.wikidata.org/sparql'

# Pageview API endpoint
pageview_url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/all-agents/{title}/monthly/{start}/{end}'

# Request headers
headers = {
    'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)',
    "Accept": "application/json"
}

# Count of each entity


# Split languages into smaller batches
batch_size = 5
for cat in categories:
    all_sitelinks = {}
    all_page_titles = {}
    entity_count = {}
    for i in range(0, len(languages), batch_size):
        batch = languages[i:i + batch_size]

        query = """
            SELECT ?entity """ + " ".join([f"?sitelink_{lang} ?label_{lang}" for lang in batch]) + """ WHERE {
            ?entity wdt:P31 wd:""" + categories[cat] + """ .
            """ + "\n".join([f"""
                OPTIONAL {{
                    ?sitelink_{lang} schema:about ?entity ;
                                    schema:isPartOf <https://{lang}.wikipedia.org/> .
                    OPTIONAL {{ ?entity rdfs:label ?label_{lang} . FILTER(LANG(?label_{lang}) = '{lang}') }}
                }}""" for lang in batch]) + """
                FILTER(""" + " && ".join([f"(BOUND(?label_{lang}))" for lang in batch]) + """)
            }
            LIMIT 800
        """

        # Make the request to the Wikidata API
        response = requests.get(wikidata_url, params={'query': query}, headers=headers)
        response.raise_for_status()

        # Parse the response JSON
        data = response.json()

        # Update the count of each entity
        for result in data['results']['bindings']:
            entity = result['entity']['value']
            entity_count[entity] = entity_count.get(entity, 0) + 1
            entity_sitelinks = {
                lang: result.get(f"sitelink_{lang}", {}).get('value')
                for lang in batch
            }
            page_titles = {
                key: entity_sitelinks[key].split('/')[-1]
                    for key in entity_sitelinks}
            if entity in all_page_titles:
                all_page_titles[entity].update(page_titles)
            else:
                all_page_titles[entity] = page_titles

            if entity in all_sitelinks:
                all_sitelinks[entity].update(entity_sitelinks)
            else:
                all_sitelinks[entity] = entity_sitelinks


    # Filter entities that are available in all languages
    final_entities = [entity_tuple for entity_tuple, count in entity_count.items() if count > len(languages) // batch_size]

    all_page_titles = {entity: all_page_titles[entity] for entity in final_entities}
    all_sitelinks = {entity: all_sitelinks[entity] for entity in final_entities}

    print(len(final_entities))
    with open(f'final_wiki_data/{cat}_terms_{len(final_entities)}.json', 'w', encoding='utf-8') as file:
        json.dump(all_page_titles, file, ensure_ascii=False, indent=4)
    if len(all_page_titles) > 0 :
        # Count of page views for each entity
        page_views_count = {}

        # Retrieve page views for each entity
        for entity in all_page_titles:
            # Iterate through languages and retrieve entities

            for lang in languages:
                if lang not in page_views_count:
                    page_views_count[lang] = {}

                page_title = all_page_titles[entity][lang]
                
                # Make the request to the Pageview API
                pv_response = requests.get(
                    pageview_url.format(project=f"{lang}.wikipedia", title=page_title, start="20190601", end="20230601"),
                    headers=headers
                )
                
                # Parse the response JSON and add page views to the count
                pv_data = pv_response.json()
                if 'items' in pv_data:
                    for item in pv_data['items']:
                        if entity in page_views_count[lang]:
                            page_views_count[lang][entity] += item['views']
                        else:
                            page_views_count[lang][entity] = item['views']

        # Sort entities by page views
        sorted_entities = {lang: sorted(page_views_count[lang], key=lambda x: x[1], reverse=True) for lang in languages}

        for lang in languages:
            for i in range(len(sorted_entities[lang])):
                entity = sorted_entities[lang][i]
                sorted_entities[lang][i] = [entity, page_views_count[lang][entity], all_page_titles[entity][lang]]

        # Save the results to a local file in JSON format
        with open(f'final_wiki_data/{cat}_by_page_views.json', 'w', encoding='utf-8') as file:
            json.dump(sorted_entities, file, ensure_ascii=False, indent=4)