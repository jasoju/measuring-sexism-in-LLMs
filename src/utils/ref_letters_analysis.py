# code based on: https://github.com/uclanlp/biases-llm-reference-letters/blob/main/biases_string_matching.py

import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import utils.word_constants as word_constants

def count_words(texts, word_patterns):
    """
    Counts occurrences of words in `texts` matching the given `word_patterns`.
    """
    counts = {key: 0 for key in word_patterns.keys()}
    total_words = 0

    for text in texts:
        words = text.split()
        total_words += len(words)
        for word in words:
            for category, pattern in word_patterns.items():
                if pattern.search(word):
                    counts[category] += 1

    return counts, total_words

    
def analyze_ref_letters(df):

    ref_letters_m = df[df['gender'] == 'male']["response"].str.lower().tolist()
    ref_letters_f = df[df['gender'] == 'female']["response"].str.lower().tolist()

    # precompile regex patterns
    word_patterns = {key: re.compile(r'\b(' + '|'.join(words) + r')\b', re.IGNORECASE)
                     for key, words in {
                         'ability': word_constants.ability_words,
                         'standout': word_constants.standout_words,
                         'agentic': word_constants.agentic_words,
                         'communal': word_constants.communal_words,
                         'grindstone': word_constants.grindstone_words,
                     }.items()}

    # process male and female letters at the same time
    with ThreadPoolExecutor() as executor:
        future_m = executor.submit(count_words, ref_letters_m, word_patterns)
        future_f = executor.submit(count_words, ref_letters_f, word_patterns)

        counts_m, total_words_m = future_m.result()
        counts_f, total_words_f = future_f.result()

    # assign categories to male/female
    male_categories = ['ability', 'standout', 'agentic']
    female_categories = ['communal', 'grindstone']

    # calculate scores and print results
    small_number = 0.001
    results = {}
    for category in word_patterns.keys():
        male_count = counts_m[category]
        female_count = counts_f[category]

        male_ratio = (male_count + small_number) / (total_words_m - male_count + small_number)
        female_ratio = (female_count + small_number) / (total_words_f - female_count + small_number)

        # compute score based on category type
        if category in male_categories:
            score = male_ratio / female_ratio
        elif category in female_categories:
            score = female_ratio / male_ratio
        else: 
            raise ValueError

        results[f"{category}_male_count"] = male_count
        results[f"{category}_female_count"] = female_count
        results[f"{category}_OR"] = score
    
    return pd.Series(results)