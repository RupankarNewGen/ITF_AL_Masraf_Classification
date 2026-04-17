
import re
from fuzzywuzzy import fuzz
from geonamescache import GeonamesCache
from difflib import get_close_matches
import pycountry


def remove_symbols(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()  # Strip spaces from the start and end of the text
    return cleaned_text

def extract_country_from_text(text):
    text = remove_symbols(text)
    gc = GeonamesCache()
    countries = gc.get_countries_by_names()
    country_names = [country["name"] for country in countries.values()]
    country_names += [country.alpha_3 for country in pycountry.countries]
    country_names += [country.alpha_2 for country in pycountry.countries]
    country_names.append("UAE")
    matching_countries = []
    for country in country_names:
        if fuzz.partial_ratio(country.lower(), text.lower()) >= 60:
            matching_countries.append(country.lower() if len(country.lower())>2 else "")

            
    words = re.findall(r'\b\w+\b', text)
    address_list = []
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            # Combine consecutive words
            combined_words = ' '.join(words[i:j])
            if combined_words.lower() in matching_countries:
                #print('combined_words', combined_words)     
                address_list.append(combined_words)
    return address_list
