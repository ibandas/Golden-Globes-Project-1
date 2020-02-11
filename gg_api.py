'''Version 0.35'''
import nltk
from nltk.corpus import names
from nltk.tree import Tree
import json
from collections import defaultdict
from datetime import datetime, timedelta
from imdb import IMDb, IMDbError
import unicodedata
import re
import time
from fuzzywuzzy import fuzz  # requires python-Levenshtein or it will be terribly slow

nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('names')

# sr = ['Golden', 'Globes', 'golden', 'I', 'globes', 'Globe', 'The', '#GoldenGlobes', 'globe', '-', 'Hollywood',
#       'Globes.', 'GOLDEN', '2020', '2020:', '&amp', '#GoldenGIobes', '#goldenglobes',
#       'host', 'Host', 'hosts', 'ever', '.', ':', '!', ',', '?']
# extra_sr = stopwords.words('english')
#
# sr.extend(extra_sr)

male_names = names.words('male.txt')
female_names = names.words('female.txt')

OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama',
                        'best performance by an actress in a motion picture - drama',
                        'best performance by an actor in a motion picture - drama',
                        'best motion picture - comedy or musical',
                        'best performance by an actress in a motion picture - comedy or musical',
                        'best performance by an actor in a motion picture - comedy or musical',
                        'best animated feature film', 'best foreign language film',
                        'best performance by an actress in a supporting role in a motion picture',
                        'best performance by an actor in a supporting role in a motion picture',
                        'best director - motion picture', 'best screenplay - motion picture',
                        'best original score - motion picture', 'best original song - motion picture',
                        'best television series - drama',
                        'best performance by an actress in a television series - drama',
                        'best performance by an actor in a television series - drama',
                        'best television series - comedy or musical',
                        'best performance by an actress in a television series - comedy or musical',
                        'best performance by an actor in a television series - comedy or musical',
                        'best mini-series or motion picture made for television',
                        'best performance by an actress in a mini-series or motion picture made for television',
                        'best performance by an actor in a mini-series or motion picture made for television',
                        'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television',
                        'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - comedy or musical',
                        'best performance by an actress in a motion picture - drama',
                        'best performance by an actor in a motion picture - drama',
                        'best performance by an actress in a motion picture - comedy or musical',
                        'best performance by an actor in a motion picture - comedy or musical',
                        'best performance by an actress in a supporting role in any motion picture',
                        'best performance by an actor in a supporting role in any motion picture',
                        'best director - motion picture', 'best screenplay - motion picture',
                        'best motion picture - animated', 'best motion picture - foreign language',
                        'best original score - motion picture', 'best original song - motion picture',
                        'best television series - drama', 'best television series - comedy or musical',
                        'best television limited series or motion picture made for television',
                        'best performance by an actress in a limited series or a motion picture made for television',
                        'best performance by an actor in a limited series or a motion picture made for television',
                        'best performance by an actress in a television series - drama',
                        'best performance by an actor in a television series - drama',
                        'best performance by an actress in a television series - comedy or musical',
                        'best performance by an actor in a television series - comedy or musical',
                        'best performance by an actress in a supporting role in a series, limited series or motion picture made for television',
                        'best performance by an actor in a supporting role in a series, limited series or motion picture made for television',
                        'cecil b. demille award']

# Checks/Incorrects for 2013
awards_regex = {
    'best motion picture - drama': r"^(?=.*\bbest\b)(?=.*\bdrama\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    'best motion picture - comedy or musical': r"^(?=.*\bbest\b)(?=.*\b(comedy|musical)\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|drama).)*$",
    # # Incorrect (Came fourth after GoldenGlobes, Jennifer Lawrence, and Julianne Moore) - Second time, nothing came up
    'best performance by an actress in a motion picture - drama': r"^(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\bdrama\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",
    'best performance by an actor in a motion picture - drama': r"^(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\bdrama\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",
    'best performance by an actress in a motion picture - comedy or musical': r"^(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\bcomedy\b)(?=.*\bmusical\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",

    'best performance by an actor in a motion picture - comedy or musical': r"^(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\bcomedy\b)(?=.*\bmusical\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",
    'best performance by an actress in a supporting role in any motion picture': r"^(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\bsupporting\b)(?=.*\b\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",
    'best performance by an actor in a supporting role in any motion picture': r"^(?=.*\b(best|win.*)\b)(?=.*\bactor\b)(?=.*\b(support.*|role)\b)(?=.*\b\b)(?=.*\b(movie|motion)\b)^((?!television|tv|TV).)*$",
    'best director - motion picture': r"^(?=.*\bbest\b)(?=.*\bdirector\b)(?=.*\b\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",
    'best screenplay - motion picture': r"^(?=.*\bbest\b)(?=.*\bscreenplay\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    # # CHECK (Brave)
    'best motion picture - animated': r"^(?=.*\bbest\b)(?=.*\b(animat.*)\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    # Incorrect (the people came up, but not the movie 'Amour' itself)
    'best motion picture - foreign language': r"^(?=.*\bbest\b)(?=.*\b(foreign|language)\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    # Incorrect (but not by much, will work once only considers name)
    'best original score - motion picture': r"^(?=.*\bbest\b)(?=.*\b(original score)\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    # CHECK (Skyfall - once names and golden globes are taken out)
    'best original song - motion picture': r"^(?=.*\bbest\b)(?=.*\b(original song)\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    # CHECK (ONCE GOLDENGLOBES IS TAKEN OUT) (Second time nothing came up)
    'best television series - drama': r"^(?=.*\bbest\b)(?=.*\bdrama\b)(?=.*\b(television|tv|TV)\b)^((?!movie|motion|picture|comedy|musical).)*$",
    # CHECK (Once golden globes and people are taken out) - Doesn't come up on the second go around
    'best television series - comedy or musical': r"^(?=.*\bbest\b)(?=.*\b(comedy|musical)\b)(?=.*\b(television|tv|TV)\b)^((?!movie|motion|picture|drama).)*$",
    # Incorrect (Didn't come up, very low frequencies all around)
    'best television limited series or motion picture made for television': r"^(?=.*\bbest\b)(?=.*\b(tv|television)\b)(?=.*\b(mini|limited|motion|picture|series|movie)\b)^((?!drama).)*$",
    'best performance by an actress in a limited series or a motion picture made for television': r"^(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\b(tv|television)\b)(?=.*\b(mini|limited|motion|picture|series)\b)^((?!drama).)*$",
    # # Incorrect (nothing came up)
    'best performance by an actor in a limited series or a motion picture made for television': r"^(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\b(tv|television)\b)(?=.*\b(limited|motion|picture|series)\b).*$",
    # # Incorrect (nothing came up)
    'best performance by an actress in a television series - drama': r"^(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\bdrama\b)(?=.*\b(tv|television)\b)(?=.*\b(series)\b).*$",
    # # CHECK (Once only names are considered) - Close tie on second run (Damien Lewis)
    'best performance by an actor in a television series - drama': r"^(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\bdrama\b)(?=.*\b(tv|television)\b)(?=.*\b(series)\b).*$",
    'best performance by an actress in a television series - comedy or musical': r"^(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\b(comedy|musical)\b)(?=.*\b(tv|television)\b)(?=.*\b(series)\b).*$",
    'best performance by an actor in a television series - comedy or musical': r"^(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\b(comedy|musical)\b)(?=.*\b(tv|television)\b)(?=.*\b(series)\b).*$",
    'best performance by an actress in a supporting role in a series, limited series or motion picture made for television': r"^(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\b(support.*|role)\b)(?=.*\b(tv|television)\b)(?=.*\b(motion|picture)\b)(?=.*\b(mini|limited|series)\b)^((?!drama).)*$",
    'best performance by an actor in a supporting role in a series, limited series or motion picture made for television': r"^(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\b(support.*|role)\b)(?=.*\b(tv|television)\b)(?=.*\b(motion|picture)\b)(?=.*\b(mini|limited|series)\b)^((?!drama).)*$",
    'cecil b. demille award': r"^(?=.*\bcecil\b)(?=.*\bdemille\b)(?=.*\b\b)(?=.*\baward\b).*$"
}


presenters_regex = {
    'best motion picture - drama': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bdrama\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    'best motion picture - comedy or musical': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\b(comedy|musical)\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|drama).)*$",
    # # Incorrect (Came fourth after GoldenGlobes, Jennifer Lawrence, and Julianne Moore) - Second time, nothing came up
    'best performance by an actress in a motion picture - drama': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\bdrama\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",
    'best performance by an actor in a motion picture - drama': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\bdrama\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",
    'best performance by an actress in a motion picture - comedy or musical': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\bcomedy\b)(?=.*\bmusical\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",

    'best performance by an actor in a motion picture - comedy or musical': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\bcomedy\b)(?=.*\bmusical\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",
    'best performance by an actress in a supporting role in any motion picture': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\bsupporting\b)(?=.*\b\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",
    'best performance by an actor in a supporting role in any motion picture': r"^(?=.*\bpresent.*\b)(?=.*\b(best|win.*)\b)(?=.*\bactor\b)(?=.*\b(support.*|role)\b)(?=.*\b\b)(?=.*\b(movie|motion)\b)^((?!television|tv|TV).)*$",
    'best director - motion picture': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bdirector\b)(?=.*\b\b)(?=.*\bmotion\b)(?=.*\bpicture\b).*$",
    'best screenplay - motion picture': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bscreenplay\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    # # CHECK (Brave)
    'best motion picture - animated': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\b(animat.*)\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    # Incorrect (the people came up, but not the movie 'Amour' itself)
    'best motion picture - foreign language': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\b(foreign|language)\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    # Incorrect (but not by much, will work once only considers name)
    'best original score - motion picture': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\b(original score)\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    # CHECK (Skyfall - once names and golden globes are taken out)
    'best original song - motion picture': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\b(original song)\b)(?=.*\b(motion|picture|movie)\b)^((?!television|tv|TV|comedy|musical).)*$",
    # CHECK (ONCE GOLDENGLOBES IS TAKEN OUT) (Second time nothing came up)
    'best television series - drama': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bdrama\b)(?=.*\b(television|tv|TV)\b)^((?!movie|motion|picture|comedy|musical).)*$",
    # CHECK (Once golden globes and people are taken out) - Doesn't come up on the second go around
    'best television series - comedy or musical': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\b(comedy|musical)\b)(?=.*\b(television|tv|TV)\b)^((?!movie|motion|picture|drama).)*$",
    # Incorrect (Didn't come up, very low frequencies all around)
    'best television limited series or motion picture made for television': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\b(tv|television)\b)(?=.*\b(mini|limited|motion|picture|series|movie)\b)^((?!drama).)*$",
    'best performance by an actress in a limited series or a motion picture made for television': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\b(tv|television)\b)(?=.*\b(mini|limited|motion|picture|series)\b)^((?!drama).)*$",
    # # Incorrect (nothing came up)
    'best performance by an actor in a limited series or a motion picture made for television': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\b(tv|television)\b)(?=.*\b(limited|motion|picture|series)\b).*$",
    # # Incorrect (nothing came up)
    'best performance by an actress in a television series - drama': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\bdrama\b)(?=.*\b(tv|television)\b)(?=.*\b(series)\b).*$",
    # # CHECK (Once only names are considered) - Close tie on second run (Damien Lewis)
    'best performance by an actor in a television series - drama': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\bdrama\b)(?=.*\b(tv|television)\b)(?=.*\b(series)\b).*$",
    'best performance by an actress in a television series - comedy or musical': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\b(comedy|musical)\b)(?=.*\b(tv|television)\b)(?=.*\b(series)\b).*$",
    'best performance by an actor in a television series - comedy or musical': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\b(comedy|musical)\b)(?=.*\b(tv|television)\b)(?=.*\b(series)\b).*$",
    'best performance by an actress in a supporting role in a series, limited series or motion picture made for television': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactress\b)(?=.*\b(support.*|role)\b)(?=.*\b(tv|television)\b)(?=.*\b(motion|picture)\b)(?=.*\b(mini|limited|series)\b)^((?!drama).)*$",
    'best performance by an actor in a supporting role in a series, limited series or motion picture made for television': r"^(?=.*\bpresent.*\b)(?=.*\bbest\b)(?=.*\bactor\b)(?=.*\b(support.*|role)\b)(?=.*\b(tv|television)\b)(?=.*\b(motion|picture)\b)(?=.*\b(mini|limited|series)\b)^((?!drama).)*$",
    'cecil b. demille award': r"^(?=.*\bpresent.*\b)(?=.*\bcecil\b)(?=.*\bdemille\b)(?=.*\b\b)(?=.*\baward\b).*$"
}


def get_continous_chunks(text):
    chunked = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for chunk in chunked:
        if type(chunk) == Tree:
            current_chunk.append(" ".join([token for token, pos in chunk.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continue
    # result = [i.encode('ascii', 'ignore').strip() for i in continuous_chunk]
    return continuous_chunk


# Complete and passes 2013/2015/2020 with flying colors
def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    with open('./data/gg{year}.json'.format(year=year)) as f:
        tweets = json.load(f)
        # tweets = [json.loads(line) for line in f]

        start = datetime.now()
        pot = timedelta(seconds=300)
        named_entities = defaultdict(int)
        for tweet in tweets:
            if start + pot < datetime.now():
                break
            tweet_text = tweet.get('text')
            # print(tweet_text)
            x = ["hosting", "hosts"]
            y = ["next", "year"]
            negative = True
            for negative_pattern in y:
                if negative_pattern not in tweet_text:
                    pass
                else:
                    negative = False
                    break
            if negative:
                for pattern in x:
                    if pattern in tweet_text:
                        tweet_named_entities = get_continous_chunks(tweet_text)
                        for ne in tweet_named_entities:
                            # print(ne)
                            named_entities[ne] += 1
                        break

    print("\n\n")
    merged = merge_keys(named_entities)
    hosts = calculate_hosts(merged)
    print(hosts)
    return hosts


# This function is to print out the most common named entities
def show_freq_hosts(ne):
    host_list = []
    freq_count = 0
    freq = nltk.FreqDist(ne)
    for key, val in freq.most_common():
        host_list.append(key)
        freq_count += 1
        print(str(key) + ': ' + str(val) + ', '),
        if freq_count == 5:
            break
    print('\n')
    return host_list


def shorten_dict(ne):
    result_dict = {}
    freq = nltk.FreqDist(ne)
    counter = 0
    for key, val in freq.most_common():
        if counter == 60:
            break
        result_dict[key] = val
        counter += 1
    return result_dict


# This function is to merge keys that are similar based off substring ('Tina' values
# merge into 'Tina Fey' values). This makes things more statistically valid
def merge_keys(ne):
    result_dict = {}
    already_viewed_terms = []
    freq = nltk.FreqDist(ne)
    for key, val in freq.most_common():
        freq_split = key.split()
        if len(freq_split) == 2 and (freq_split[0] in (female_names or male_names)) \
                and (freq_split[0] not in already_viewed_terms):
            result_dict[key] = val
            for term in freq_split:
                if term not in already_viewed_terms:
                    for k, v in freq.most_common():
                        if k != key and term in k:
                            result_dict[key] += v
                    already_viewed_terms.append(term)
    return result_dict


# This figures out how many hosts there are and returns them
def calculate_hosts(ne):
    hosts = []
    max_value_key = max(ne, key=ne.get)

    hosts.append(max_value_key)
    for k in ne:
        if k != max_value_key and (ne.get(k) >= (ne[max_value_key] * .7)):
            hosts.append(k)

    return hosts


def get_tweets(year):
    tweets = []
    try:
        with open('gg' + str(year) + '.json', 'r', encoding='utf8') as f:
            tweets = json.load(f)
    except json.JSONDecodeError:
        with open('gg' + str(year) + '.json', 'r', encoding='utf8') as f:
            for line in f:
                tweets.append(json.loads(line))
    return tweets


def dict_inc(dictionary, key):
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1
    return dictionary


def dict_sort(dictionary):
    sorted_d = sorted(dictionary.items(), reverse=True, key=lambda x: x[1])
    return sorted_d


def all_after(input_list, value):
    for i in range(len(input_list)):
        if input_list[i].lower() == value:
            return input_list[i+1:]
    return input_list


def all_before(input_list, value):
    for i in range(len(input_list)):
        if input_list[i].lower() == value:
            return input_list[:i]
    return input_list


def all_between(input_list, after_this, before_this):
    return all_after(all_before(input_list, before_this), after_this)


def concentrate(dictionary):
    copy_dict = {}
    for item1 in dictionary:
        copy_dict[item1] = dictionary[item1]
        for item2 in dictionary:
            if len(item1) > len(item2):
                ratio = fuzz.token_set_ratio(item2, item1) / 100
                to_add = ratio * ratio * dictionary[item2]
                copy_dict[item1] += to_add
    for item in copy_dict:
        copy_dict[item] = copy_dict[item] / (len(item)**(1/2))
    return copy_dict


def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    start = time.time()
    tweets = get_tweets(year)

    sr = ['golden', 'globe', 'globes', 'goldenglobes', 'goldenglobe', 'show', '2020', 'goldenglobes2020',
          'goldenglobes2015', 'nbc', 'cbr', 'click', 'award', 'experience', 'i', 'read', 'amp', 'answer',
          'follow', 'category']
    chunked_dict = {}
    chunkGram1 = r"""Chunk: {<JJS|RBS><NN><IN><DT><NN><IN><DT><NN><NN><:|,><JJ><CC><NN>|<JJS|RBS><NN><IN><DT><NN><IN><DT><NN><NN><:|,><NN>|<JJS|RBS><NN><IN><DT><NN><IN><DT><JJ><NN><IN><DT><NN><NN>|<JJS|RBS><NN><IN><DT><NN><IN><DT><JJ><NN><IN><DT><NN><,><JJ><NN><CC><NN><NN><VBN><IN><NN>|<JJS|RBS><NN><IN><DT><NN><IN><DT><JJ><NN><CC><DT><NN><NN><VBN><IN><NN>|<JJS|RBS><JJ|NN><NN><:|,><JJ><CC>?<NN>|<JJS|RBS><JJ|NN><NN><:|,><NN><NN>|<JJS|RBS><NN><NN><:|,><VBD>|<JJS|RBS><NN><:|,><NN><NN>|<JJS|RBS><NN><JJ><NN><CC><NN><NN><VBN><IN><NN>}"""

    chunkGram2 = r"""Chunk: {<JJS|RBS><NN><IN><DT><NN><IN><DT><NN><NN><:|,>?<NN><CC><JJ>|<JJS|RBS><NN><IN><DT><NN><IN><DT><NN><NN><:|,>?<NN>|<JJS|RBS><NN><IN><DT><NN><IN><DT><JJ><NN><IN><DT><NN><NN>|<JJS|RBS><NN><IN><DT><NN><IN><DT><JJ><NN><IN><DT><NN><,>?<NNS><CC><NN><NN><VBN><IN><NN>|<JJS|RBS><NN><IN><DT><NN><IN><DT><NNS><CC><NN><NN><VBN><IN><NN>|<JJS|RBS><NN><NN><:|,>?<NN><CC>?<JJ>?|<JJS|RBS><JJ|NN><NN><:|,>?<NN><NN>|<JJS|RBS><NN><:|,>?<NN><NN>|<JJS|RB|RBS><VBN|JJ><NN><NN>|<JJS|RBS><NNS><CC><NN><NN><VBN><IN><NN>}"""

    if int(year) > 2016:
        chunkGram = chunkGram1
    else:
        chunkGram = chunkGram2
    chunkParser = nltk.RegexpParser(chunkGram)
    for i in range(min(len(tweets), 1000000)):
        tweet = tweets[i]['text']
        tweet_l = tweet.lower()
        tokenized = re.findall(r"\w+-\w+|\w+|-", tweet_l)
        tokenized = all_before(all_before(tokenized, 'https'), 'http')
        for j in range(len(tokenized)):
            if tokenized[j] == 'tv':
                tokenized[j] = 'television'
        tokenized = [w for w in tokenized if w.lower() not in sr]

        word = ''
        if 'wins' in tokenized:
            word = 'wins'
        elif 'gets' in tokenized:
            word = 'gets'
        elif 'goes' in tokenized:
            word = 'goes'
        elif 'nominated' in tokenized:
            word = 'nominated'
        elif 'nominees' in tokenized:
            word = 'nominees'
        after_wins = ['']
        if word in ['wins', 'gets']:
            after_wins = all_after(tokenized, word)
        elif word in ['goes', 'nominees']:
            after_wins = all_before(tokenized, word)
        elif word == 'nominated':
            after_wins = all_after(tokenized, word)
            if len(after_wins) >= 2 and after_wins[0] == 'for':
                after_wins = after_wins[1:]

        if after_wins and after_wins != ['']:
            chunked = chunkParser.parse(nltk.pos_tag(after_wins))
            for chunk in chunked:
                if str(chunk)[0:6] == '(Chunk':
                    sentence = ' '.join([chonk[0] for chonk in chunk])
                    if sentence[0] == "'":
                        sentence = sentence[1:]
                    chunked_dict = dict_inc(chunked_dict, sentence.lower())
    print(time.time() - start)
    chunked_dict = concentrate(chunked_dict)
    sorted_d = sorted(chunked_dict.items(), key=lambda x: x[1])
    to_delete = []
    for i in range(len(sorted_d) - 1):
        for j in range(i + 1, len(sorted_d)):
            if fuzz.ratio(sorted_d[i][0], sorted_d[j][0]) >= 98:
                to_delete.append(i)
    new_d = [sorted_d[i] for i in range(len(sorted_d)) if i not in to_delete]
    new_d.reverse()
    highest = new_d[0][1]
    cutoff = highest * 0.55
    if highest > 1000:
        cutoff = .8 * highest
    elif highest > 250:
        cutoff = .75 * highest
    elif highest > 200:
        cutoff = .7 * highest
    elif highest > 50:
        cutoff = .65 * highest
    elif highest > 20:
        cutoff = .5 * highest
    elif highest > 10:
        cutoff = .4 * highest
    elif highest < 10:
        cutoff = .11 * highest
    else:
        cutoff = .5 * highest

    awardds = [awardd[0] for awardd in new_d if awardd[0] and awardd[1] >= cutoff]
    print(len(awardds))
    print('Finished in', (time.time() - start))
    return awardds


def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''

    # Go through each tweet. For each tweet look for each award.
    # Build a dictionary for each award.
    # Frequency map similar to hosts for each award.
    # Then take the most frequent named entities from each award and
    # put it in a list as a value for they key of the award
    return {
        "best screenplay - motion picture": ["zero dark thirty", "lincoln", "silver linings playbook", "argo"],
        "best director - motion picture": ["kathryn bigelow", "ang lee", "steven spielberg", "quentin tarantino"],
        "best performance by an actress in a television series - comedy or musical": ["zooey deschanel", "tina fey",
                                                                                      "julia louis-dreyfus",
                                                                                      "amy poehler"],
        "best foreign language film": ["the intouchables", "kon tiki", "a royal affair", "rust and bone"],
        "best performance by an actor in a supporting role in a motion picture": ["alan arkin", "leonardo dicaprio",
                                                                                  "philip seymour hoffman",
                                                                                  "tommy lee jones"],
        "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television": [
            "hayden panettiere", "archie panjabi", "sarah paulson", "sofia vergara"],
        "best motion picture - comedy or musical": ["the best exotic marigold hotel", "moonrise kingdom",
                                                    "salmon fishing in the yemen", "silver linings playbook"],
        "best performance by an actress in a motion picture - comedy or musical": ["emily blunt", "judi dench",
                                                                                   "maggie smith", "meryl streep"],
        "best mini-series or motion picture made for television": ["the girl", "hatfields & mccoys", "the hour",
                                                                   "political animals"],
        "best original score - motion picture": ["argo", "anna karenina", "cloud atlas", "lincoln"],
        "best performance by an actress in a television series - drama": ["connie britton", "glenn close",
                                                                          "michelle dockery", "julianna margulies"],
        "best performance by an actress in a motion picture - drama": ["marion cotillard", "sally field",
                                                                       "helen mirren", "naomi watts", "rachel weisz"],
        "cecil b. demille award": [],
        "best performance by an actor in a motion picture - comedy or musical": ["jack black", "bradley cooper",
                                                                                 "ewan mcgregor", "bill murray"],
        "best motion picture - drama": ["django unchained", "life of pi", "lincoln", "zero dark thirty"],
        "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television": [
            "max greenfield", "danny huston", "mandy patinkin", "eric stonestreet"],
        "best performance by an actress in a supporting role in a motion picture": ["amy adams", "sally field",
                                                                                    "helen hunt", "nicole kidman"],
        "best television series - drama": ["boardwalk empire", "breaking bad", "downton abbey (masterpiece)",
                                           "the newsroom"],
        "best performance by an actor in a mini-series or motion picture made for television": ["benedict cumberbatch",
                                                                                                "woody harrelson",
                                                                                                "toby jones",
                                                                                                "clive owen"],
        "best performance by an actress in a mini-series or motion picture made for television": ["nicole kidman",
                                                                                                  "jessica lange",
                                                                                                  "sienna miller",
                                                                                                  "sigourney weaver"],
        "best animated feature film": ["frankenweenie", "hotel transylvania", "rise of the guardians",
                                       "wreck-it ralph"],
        "best original song - motion picture": ["act of valor", "stand up guys", "the hunger games", "les miserables"],
        "best performance by an actor in a motion picture - drama": ["richard gere", "john hawkes", "joaquin phoenix",
                                                                     "denzel washington"],
        "best television series - comedy or musical": ["the big bang theory", "episodes", "modern family", "smash"],
        "best performance by an actor in a television series - drama": ["steve buscemi", "bryan cranston",
                                                                        "jeff daniels", "jon hamm"],
        "best performance by an actor in a television series - comedy or musical": ["alec baldwin", "louis c.k.",
                                                                                    "matt leblanc", "jim parsons"]
    }
    # Your code here



sub_tweet_data = []


def get_winner(year):
    # Create a constant dict of each award as the key
    # and the values are a list of regex for that award

    # Then create a dynamic dict that has each award as the key again
    # but the values are frequencyMap (similar to hosts).
    # So it's a dict nested in a dict

    awards_mapped_to_entities = {
        'best motion picture - drama': defaultdict(int),
        'best motion picture - comedy or musical': defaultdict(int),
        'best performance by an actress in a motion picture - drama': defaultdict(int),
        'best performance by an actor in a motion picture - drama': defaultdict(int),
        'best performance by an actress in a motion picture - comedy or musical': defaultdict(int),
        'best performance by an actor in a motion picture - comedy or musical': defaultdict(int),
        'best performance by an actress in a supporting role in any motion picture': defaultdict(int),
        'best performance by an actor in a supporting role in any motion picture': defaultdict(int),
        'best director - motion picture': defaultdict(int),
        'best screenplay - motion picture': defaultdict(int),
        'best motion picture - animated': defaultdict(int),
        'best motion picture - foreign language': defaultdict(int),
        'best original score - motion picture': defaultdict(int),
        'best original song - motion picture': defaultdict(int),
        'best television series - drama': defaultdict(int),
        'best television series - comedy or musical': defaultdict(int),
        'best television limited series or motion picture made for television': defaultdict(int),
        'best performance by an actress in a limited series or a motion picture made for television': defaultdict(int),
        'best performance by an actor in a limited series or a motion picture made for television': defaultdict(int),
        'best performance by an actress in a television series - drama': defaultdict(int),
        'best performance by an actor in a television series - drama': defaultdict(int),
        'best performance by an actress in a television series - comedy or musical': defaultdict(int),
        'best performance by an actor in a television series - comedy or musical': defaultdict(int),
        'best performance by an actress in a supporting role in a series, limited series or motion picture made for television': defaultdict(
            int),
        'best performance by an actor in a supporting role in a series, limited series or motion picture made for television': defaultdict(
            int),
        'cecil b. demille award': defaultdict(int)
    }

    year_minus_one = int(year) - 1
    year_minus_two = int(year) - 2
    with open('./data/gg{year}.json'.format(year=year)) as f, open('./data/rpm_{year}.json'.format(year=str(year_minus_one))) as imdb_1, open('./data/rpm_{year}.json'.format(year=str(year_minus_two))) as imdb_2:
        tweets = json.load(f)
        imdb_db_1 = json.load(imdb_1)
        imdb_db_2 = json.load(imdb_2)
        imdb_total = [imdb_db_1, imdb_db_2]

        start = datetime.now()
        pot = timedelta(seconds=300)
        for tweet in tweets:
            if start + pot < datetime.now():
                break
            tweet_text = tweet.get('text')
            sub_tweet_data.append(tweet_text)
            for award in awards_regex:
                regex_ = awards_regex.get(award)
                matches = re.findall(regex_, tweet_text.lower(), re.MULTILINE)
                if matches:
                    tweet_named_entities = get_continous_chunks(tweet_text)
                    for ne in tweet_named_entities:
                        awards_mapped_to_entities.get(award)[ne] += 1

    print("\n\n")

    result = award_winner_master(awards=awards_mapped_to_entities, db=imdb_total)
    print(result)
    return result


    #
    # '''Winners is a dictionary with the hard coded award
    # names as keys, and each entry containing a single string.
    # Do NOT change the name of this function or what it returns.'''
    # # Your code here
    # return {
    #     "best screenplay - motion picture": "Django Unchained",
    #     "best director - motion picture": "ben affleck",
    #     "best performance by an actress in a television series - comedy or musical": "lena dunham",
    #     "best foreign language film": "amour",
    #     "best performance by an actor in a supporting role in a motion picture": "christoph waltz",
    #     "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television": "maggie smith",
    #     "best motion picture - comedy or musical": "les miserables",
    #     "best performance by an actress in a motion picture - comedy or musical": "jennifer lawrence",
    #     "best mini-series or motion picture made for television": "game change",
    #     "best original score - motion picture": "life of pi",
    #     "best performance by an actress in a television series - drama": "claire danes",
    #     "best performance by an actress in a motion picture - drama": "jessica chastain",
    #     "cecil b. demille award": "jodie foster",
    #     "best performance by an actor in a motion picture - comedy or musical": "hugh jackman",
    #     "best motion picture - drama": "argo",
    #     "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television": "ed harris",
    #     "best performance by an actress in a supporting role in a motion picture": "anne hathaway",
    #     "best television series - drama": "homeland",
    #     "best performance by an actor in a mini-series or motion picture made for television": "kevin costner",
    #     "best performance by an actress in a mini-series or motion picture made for television": "julianne moore",
    #     "best animated feature film": "brave",
    #     "best original song - motion picture": "skyfall",
    #     "best performance by an actor in a motion picture - drama": "daniel day-lewis",
    #     "best television series - comedy or musical": "girls",
    #     "best performance by an actor in a television series - drama": "damian lewis",
    #     "best performance by an actor in a television series - comedy or musical": "don cheadle"
    # }


def merge_keys_winner(ne, award_to_person=True):
    result_dict = {}
    already_viewed_terms = []
    freq = nltk.FreqDist(ne)
    for key, val in freq.most_common():
        freq_split = key.split()
        if len(freq_split) != 0:
            if len(freq_split) == 2 and (freq_split[0] in (female_names or male_names)) \
                    and (freq_split[0] not in already_viewed_terms) and award_to_person:
                result_dict[key] = val
                for term in freq_split:
                    if term not in already_viewed_terms:
                        for k, v in freq.most_common():
                            if k != key and term in k:
                                result_dict[key] += v
                                del freq[k]
                        already_viewed_terms.append(term)
            elif (freq_split[0] not in (female_names or male_names)) \
                    and (freq_split[0] not in already_viewed_terms):
                result_dict[key] = val
                for term in freq_split:
                    if term not in already_viewed_terms:
                        for k, v in freq.most_common():
                            if k != key and term in k:
                                result_dict[key] += v
                                del freq[k]
                            already_viewed_terms.append(term)

        return result_dict


def merge_keys_for_title_winner_sub(ne):
    result_dict = defaultdict(int)
    already_viewed_terms = []
    already_viewed_entities = []
    freq = nltk.FreqDist(ne)
    for key, val in freq.most_common():
        result_dict[key] = val
        freq_split = key.split()
        # Checks each part of the entity ("Amy Poehler" means "Amy" and "Poehler")
        for term in freq_split:
            if term not in already_viewed_terms:
                for k, v in freq.most_common():
                    # print("Key: {k}  Val: {v}".format(k=k, v=v))
                    if k != key and k not in already_viewed_entities and term in k:
                        already_viewed_entities.append(k)
                        result_dict[key] += v
                        # del freq[k]
                already_viewed_terms.append(term)
    return result_dict


def merge_keys_for_person_winner_sub(ne):
    result_dict = defaultdict(int)
    already_viewed_terms = []
    already_viewed_entities = []
    freq = nltk.FreqDist(ne)
    for key, val in freq.most_common():
        result_dict[key] = val
        freq_split = key.split()
        # Checks each part of the entity ("Amy Poehler" means "Amy" and "Poehler")
        for term in freq_split:
            if term not in already_viewed_terms:
                for k, v in freq.most_common():
                    # print("Key: {k}  Val: {v}".format(k=k, v=v))
                    if k != key and k not in already_viewed_entities and term in k:
                        already_viewed_entities.append(k)
                        result_dict[key] += v
                        # del freq[k]
                already_viewed_terms.append(term)
    return result_dict


def merge_keys_for_person_winner_imdb(ne, db):
    result_dict = defaultdict(int)
    freq = nltk.FreqDist(ne)
    for key, val in freq.most_common():
        # If the name is in imdb_db
        movies = []
        db_0 = db[0].get(key)
        db_1 = db[1].get(key)
        if db_0:
            movies = db[0].get(key)
            if db_1:
                movies.extend(db_1)
        elif db_1:
            movies = db[1].get(key)

        if movies:
            # Strips any accents from a movie, like 'Les Misérables' to assist in matching terms
            titles_accent_stripped = []
            for movie in movies:
                titles_accent_stripped.append(strip_accents(movie))
            # Adds the key to the result_dict
            result_dict[key] = val
            for k, v in freq.most_common():
                if k != key and k in titles_accent_stripped:
                    result_dict[key] += v
    return result_dict


def take_out_names(entities):
    result_dict = defaultdict(int)
    freq = nltk.FreqDist(entities)
    for key, val in freq.most_common():
        if not is_male(key) and not is_female(key):
            result_dict[key] = val
    return result_dict


def is_male(entity):
    entity_split = entity.split()
    if entity_split[0] in male_names:
        return True
    else:
        return False


def is_female(entity):
    entity_split = entity.split()
    if entity_split[0] in female_names:
        return True
    else:
        return False


def remove_names(entities):
    result_dict = defaultdict(int)
    freq = nltk.FreqDist(entities)
    for key, val in freq.most_common():
        if not is_female(key) and not is_male(key) and not forbidden_words(key):
            result_dict[key] = val
    return result_dict


def forbidden_words(entity):
    forb = ['golden', 'motion', 'picture', 'drama', 'comedy', 'actor', 'actress', 'globe', 'best', 'musical', 'live']
    for bidden in forb:
        if bidden in entity.lower():
            return True


def is_movie(entities, db):
    count = 0
    already_added_movies = []
    result_dict = defaultdict(int)
    freq = nltk.FreqDist(entities)
    for entity, val in freq.most_common():
        for k, v in db[0].items():
            if (entity.lower() in [x.lower() for x in v]) and (entity.lower() not in already_added_movies):
                result_dict[entity] = val
                count += 1
            if count >= 5:
                return result_dict

    return result_dict


def is_full_name(entity):
    if len(entity) == 2:
        if entity[0] in (male_names or female_names):
            return True
    else:
        return False


def strip_accents(text):

    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)


def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    awards_mapped_to_entities = {
        'best motion picture - drama': defaultdict(int),
        'best motion picture - comedy or musical': defaultdict(int),
        'best performance by an actress in a motion picture - drama': defaultdict(int),
        'best performance by an actor in a motion picture - drama': defaultdict(int),
        'best performance by an actress in a motion picture - comedy or musical': defaultdict(int),
        'best performance by an actor in a motion picture - comedy or musical': defaultdict(int),
        'best performance by an actress in a supporting role in any motion picture': defaultdict(int),
        'best performance by an actor in a supporting role in any motion picture': defaultdict(int),
        'best director - motion picture': defaultdict(int),
        'best screenplay - motion picture': defaultdict(int),
        'best motion picture - animated': defaultdict(int),
        'best motion picture - foreign language': defaultdict(int),
        'best original score - motion picture': defaultdict(int),
        'best original song - motion picture': defaultdict(int),
        'best television series - drama': defaultdict(int),
        'best television series - comedy or musical': defaultdict(int),
        'best television limited series or motion picture made for television': defaultdict(int),
        'best performance by an actress in a limited series or a motion picture made for television': defaultdict(int),
        'best performance by an actor in a limited series or a motion picture made for television': defaultdict(int),
        'best performance by an actress in a television series - drama': defaultdict(int),
        'best performance by an actor in a television series - drama': defaultdict(int),
        'best performance by an actress in a television series - comedy or musical': defaultdict(int),
        'best performance by an actor in a television series - comedy or musical': defaultdict(int),
        'best performance by an actress in a supporting role in a series, limited series or motion picture made for television': defaultdict(
            int),
        'best performance by an actor in a supporting role in a series, limited series or motion picture made for television': defaultdict(
            int),
        'cecil b. demille award': defaultdict(int)
    }

    year_minus_one = int(year) - 1
    year_minus_two = int(year) - 2
    with open('./data/rpm_{year}.json'.format(year=str(year_minus_one))) as imdb_1, open('./data/rpm_{year}.json'.format(year=str(year_minus_two))) as imdb_2:
        imdb_db_1 = json.load(imdb_1)
        imdb_db_2 = json.load(imdb_2)
        imdb_total = [imdb_db_1, imdb_db_2]
        for tweet_text in sub_tweet_data:
            print(tweet_text)
            for award in presenters_regex:
                regex_ = presenters_regex.get(award)
                matches = re.findall(regex_, tweet_text.lower(), re.MULTILINE)
                if matches:
                    tweet_named_entities = get_continous_chunks(tweet_text)
                    for ne in tweet_named_entities:
                        awards_mapped_to_entities.get(award)[ne] += 1
        for key, val in awards_mapped_to_entities.items():
            print("{award}: ".format(award=key))
            show_freq_hosts(val)

    # Your code here
    return {
        "best screenplay - motion picture": ["robert pattinson", "amanda seyfried"],
        "best director - motion picture": ["halle berry"],
        "best performance by an actress in a television series - comedy or musical": ["aziz ansari", "jason bateman"],
        "best foreign language film": ["arnold schwarzenegger", "sylvester stallone"],
        "best performance by an actor in a supporting role in a motion picture": ["bradley cooper", "kate hudson"],
        "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television": [
            "dennis quaid", "kerry washington"],
        "best motion picture - comedy or musical": ["dustin hoffman"],
        "best performance by an actress in a motion picture - comedy or musical": ["will ferrell", "kristen wiig"],
        "best mini-series or motion picture made for television": ["don cheadle", "eva longoria"],
        "best original score - motion picture": ["jennifer lopez", "jason statham"],
        "best performance by an actress in a television series - drama": ["nathan fillion", "lea michele"],
        "best performance by an actress in a motion picture - drama": ["george clooney"],
        "cecil b. demille award": ["robert downey, jr."],
        "best performance by an actor in a motion picture - comedy or musical": ["jennifer garner"],
        "best motion picture - drama": ["julia roberts"],
        "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television": [
            "kristen bell", "john krasinski"],
        "best performance by an actress in a supporting role in a motion picture": ["megan fox", "jonah hill"],
        "best television series - drama": ["salma hayek", "paul rudd"],
        "best performance by an actor in a mini-series or motion picture made for television": ["jessica alba",
                                                                                                "kiefer sutherland"],
        "best performance by an actress in a mini-series or motion picture made for television": ["don cheadle",
                                                                                                  "eva longoria"],
        "best animated feature film": ["sacha baron cohen"],
        "best original song - motion picture": ["jennifer lopez", "jason statham"],
        "best performance by an actor in a motion picture - drama": ["george clooney"],
        "best television series - comedy or musical": ["jimmy fallon", "jay leno"],
        "best performance by an actor in a television series - drama": ["salma hayek", "paul rudd"],
        "best performance by an actor in a television series - comedy or musical": ["lucy liu", "debra messing"]
    }


def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    print("Pre-ceremony processing complete.")
    return


def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    get_winner(2013)
    # get_presenters(2013)
    # get_hosts(2013)
    return


def only_male_entities(entities):
    result_dict = defaultdict(int)
    for key, val in entities.items():
        if is_male(key):
            result_dict[key] = val
    return result_dict


def only_female_entities(entities):
    result_dict = defaultdict(int)
    for key, val in entities.items():
        if is_female(key):
            result_dict[key] = val
    return result_dict


def award_winner_master(awards, db):
    result_dict = defaultdict(str)
    result_dict['best motion picture - drama'] = award_winner_1(entities=awards.get('best motion picture - drama'), db=db)
    result_dict['best motion picture - comedy or musical'] = award_winner_2(entities=awards.get('best motion picture - comedy or musical'), db=db)
    result_dict['best performance by an actress in a motion picture - drama'] = award_winner_3(entities=awards.get('best performance by an actress in a motion picture - drama'), db=db)
    result_dict['best performance by an actor in a motion picture - drama'] = award_winner_4(entities=awards.get('best performance by an actor in a motion picture - drama'), db=db)
    result_dict['best performance by an actress in a motion picture - comedy or musical'] = award_winner_5(entities=awards.get('best performance by an actress in a motion picture - comedy or musical'), db=db)
    result_dict['best performance by an actor in a motion picture - comedy or musical'] = award_winner_6(entities=awards.get('best performance by an actor in a motion picture - comedy or musical'), db=db)
    result_dict['best performance by an actress in a supporting role in any motion picture'] = award_winner_7(entities=awards.get('best performance by an actress in a supporting role in any motion picture'), db=db)
    result_dict['best performance by an actor in a supporting role in any motion picture'] = award_winner_8(entities=awards.get('best performance by an actor in a supporting role in any motion picture'), db=db)
    result_dict['best director - motion picture'] = award_winner_9(entities=awards.get('best director - motion picture'), db=db)
    result_dict['best screenplay - motion picture']= award_winner_10(entities=awards.get('best screenplay - motion picture'), db=db)
    result_dict['best motion picture - animated'] = award_winner_11(entities=awards.get('best motion picture - animated'), db=db)
    result_dict['best motion picture - foreign language'] = award_winner_12(entities=awards.get('best motion picture - foreign language'), db=db)
    result_dict['best original score - motion picture'] = award_winner_13(entities=awards.get('best original score - motion picture'), db=db)
    result_dict['best original song - motion picture'] = award_winner_14(entities=awards.get('best original song - motion picture'), db=db)
    result_dict['best television series - drama'] = award_winner_15(entities=awards.get('best television series - drama'), db=db)
    result_dict['best television series - comedy or musical'] = award_winner_16(entities=awards.get('best television series - comedy or musical'), db=db)
    result_dict['best television limited series or motion picture made for television'] = award_winner_17(entities=awards.get('best television limited series or motion picture made for television'), db=db)
    result_dict['best performance by an actress in a limited series or a motion picture made for television'] = award_winner_18(entities=awards.get('best performance by an actress in a limited series or a motion picture made for television'), db=db)
    result_dict['best performance by an actor in a limited series or a motion picture made for television'] = award_winner_19(entities=awards.get('best performance by an actor in a limited series or a motion picture made for television'), db=db)
    result_dict['best performance by an actress in a television series - drama'] = award_winner_20(entities=awards.get('best performance by an actress in a television series - drama'), db=db)
    result_dict['best performance by an actor in a television series - drama'] = award_winner_21(entities=awards.get('best performance by an actor in a television series - drama'), db=db)
    result_dict['best performance by an actress in a television series - comedy or musical'] = award_winner_22(entities=awards.get('best performance by an actress in a television series - comedy or musical'), db=db)
    result_dict['best performance by an actor in a television series - comedy or musical'] = award_winner_23(entities=awards.get('best performance by an actor in a television series - comedy or musical'), db=db)
    result_dict['best performance by an actress in a supporting role in a series, limited series or motion picture made for television'] = award_winner_24(entities=awards.get('best performance by an actress in a supporting role in a series, limited series or motion picture made for television'), db=db)
    result_dict['best performance by an actor in a supporting role in a series, limited series or motion picture made for television'] = award_winner_25(entities=awards.get('best performance by an actor in a supporting role in a series, limited series or motion picture made for television'), db=db)
    result_dict['cecil b. demille award'] = award_winner_26(entities=awards.get('cecil b. demille award'), db=db)
    return result_dict


def remove_forbbiden(entities):
    result_dict = defaultdict(int)
    freq = nltk.FreqDist(entities)
    for key, val in freq.most_common():
        if not forbidden_words(key):
            result_dict[key] = val
    return result_dict


def remove_one_word_names(entities):
    max_ = max(entities, key=entities.get)
    if ' ' not in max_:
        del entities[max_]
        max_ = max(entities, key=entities.get)
    return max_


# best motion picture - drama
def award_winner_1(entities, db):
    print("{award}: ".format(award='best motion picture - drama'))
    shorten = shorten_dict(entities)
    names_removed = remove_names(shorten)
    sub = merge_keys_for_person_winner_sub(names_removed)
    movies = is_movie(entities=sub, db=db)
    show_freq_hosts(movies)
    return max(movies, key=movies.get)


# best motion picture - comedy or musical
def award_winner_2(entities, db):
    print("{award}: ".format(award='best motion picture - comedy or musical'))
    shorten = shorten_dict(entities)
    names_removed = remove_names(shorten)
    sub = merge_keys_for_person_winner_sub(names_removed)
    movies = is_movie(entities=sub, db=db)
    show_freq_hosts(movies)
    return max(movies, key=movies.get)


# best performance by an actress in a motion picture - drama
def award_winner_3(entities, db):
    print("{award}: ".format(award='best performance by an actress in a motion picture - drama'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    females_only = only_female_entities(merged)
    show_freq_hosts(females_only)
    name = remove_one_word_names(entities=females_only)
    return name


# best performance by an actor in a motion picture - drama
def award_winner_4(entities, db):
    print("{award}: ".format(award='best performance by an actor in a motion picture - drama'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    males_only = only_male_entities(merged)
    show_freq_hosts(males_only)
    name = remove_one_word_names(entities=males_only)
    return name


# best performance by an actress in a motion picture - comedy or musical
def award_winner_5(entities, db):
    print("{award}: ".format(award='best performance by an actress in a motion picture - comedy or musical'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    females_only = only_female_entities(merged)
    show_freq_hosts(females_only)
    name = remove_one_word_names(entities=females_only)
    return name


# best performance by an actor in a motion picture - comedy or musical
def award_winner_6(entities, db):
    print("{award}: ".format(award='best performance by an actor in a motion picture - comedy or musical'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    males_only = only_male_entities(merged)
    show_freq_hosts(males_only)
    name = remove_one_word_names(entities=males_only)
    return name


# best performance by an actress in a supporting role in any motion picture
def award_winner_7(entities, db):
    print("{award}: ".format(award='best performance by an actress in a supporting role in any motion picture'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    females_only = only_female_entities(merged)
    show_freq_hosts(females_only)
    name = remove_one_word_names(entities=females_only)
    return name


# best performance by an actor in a supporting role in any motion picture
def award_winner_8(entities, db):
    print("{award}: ".format(award='best performance by an actor in a supporting role in any motion picture'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    males_only = only_male_entities(merged)
    show_freq_hosts(males_only)
    name = remove_one_word_names(entities=males_only)
    return name


# Best Direction - Motion Picture (Correct)
def award_winner_9(entities, db):
    print("{award}: ".format(award='best director - motion picture'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    show_freq_hosts(merged)
    return max(merged, key=merged.get)


# 'best screenplay - motion picture'
def award_winner_10(entities, db):
    print("{award}: ".format(award='best screenplay - motion picture'))
    shorten = shorten_dict(entities)
    names_removed = remove_names(shorten)
    sub = merge_keys_for_person_winner_sub(names_removed)
    movies = is_movie(entities=sub, db=db)
    show_freq_hosts(movies)
    return max(movies, key=movies.get)


# best motion picture - animated
def award_winner_11(entities, db):
    print("{award}: ".format(award='best motion picture - animated'))
    shorten = shorten_dict(entities)
    names_removed = remove_names(shorten)
    sub = merge_keys_for_person_winner_sub(names_removed)
    movies = is_movie(entities=sub, db=db)
    show_freq_hosts(movies)
    return max(movies, key=movies.get)


# best motion picture - foreign language
def award_winner_12(entities, db):
    print("{award}: ".format(award='best motion picture - foreign language'))
    shorten = shorten_dict(entities)
    names_removed = remove_names(shorten)
    sub = merge_keys_for_person_winner_sub(names_removed)
    movies = is_movie(entities=sub, db=db)
    show_freq_hosts(movies)
    return max(movies, key=movies.get)


# best original score - motion picture
def award_winner_13(entities, db):
    print("{award}: ".format(award='best original score - motion picture'))
    shorten = shorten_dict(entities)
    names_removed = remove_names(shorten)
    sub = merge_keys_for_person_winner_sub(names_removed)
    movies = is_movie(entities=sub, db=db)
    show_freq_hosts(movies)
    return max(movies, key=movies.get)


# best original song - motion picture
def award_winner_14(entities, db):
    print("{award}: ".format(award='best original song - motion picture'))
    shorten = shorten_dict(entities)
    names_removed = remove_names(shorten)
    sub = merge_keys_for_person_winner_sub(names_removed)
    movies = is_movie(entities=sub, db=db)
    show_freq_hosts(movies)
    return max(movies, key=movies.get)


# best television series - drama
def award_winner_15(entities, db):
    print("{award}: ".format(award='best television series - drama'))
    shorten = shorten_dict(entities)
    names_removed = remove_names(shorten)
    sub = merge_keys_for_person_winner_sub(names_removed)
    movies = is_movie(entities=sub, db=db)
    show_freq_hosts(movies)
    return max(movies, key=movies.get)


# best television series - comedy or musical
def award_winner_16(entities, db):
    print("{award}: ".format(award='best television series - comedy or musical'))
    shorten = shorten_dict(entities)
    names_removed = remove_names(shorten)
    sub = merge_keys_for_person_winner_sub(names_removed)
    movies = is_movie(entities=sub, db=db)
    show_freq_hosts(movies)
    return max(movies, key=movies.get)


# best television limited series or motion picture made for television
def award_winner_17(entities, db):
    print("{award}: ".format(award='best television limited series or motion picture made for television'))
    shorten = shorten_dict(entities)
    names_removed = remove_names(shorten)
    sub = merge_keys_for_person_winner_sub(names_removed)
    movies = is_movie(entities=sub, db=db)
    show_freq_hosts(movies)
    return max(movies, key=movies.get)


# TODO: Change to movie as winner (maybe, wait for Viktor response)
# best performance by an actress in a limited series or a motion picture made for television
def award_winner_18(entities, db):
    print("{award}: ".format(award='best performance by an actress in a limited series or a motion picture made for television'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    females_only = only_female_entities(merged)
    show_freq_hosts(females_only)
    name = remove_one_word_names(entities=females_only)
    return name


# best performance by an actor in a limited series or a motion picture made for television
def award_winner_19(entities, db):
    print("{award}: ".format(award='best performance by an actor in a limited series or a motion picture made for television'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    males_only = only_male_entities(merged)
    show_freq_hosts(males_only)
    name = remove_one_word_names(entities=males_only)
    return name


# best performance by an actress in a television series - drama
def award_winner_20(entities, db):
    print("{award}: ".format(award='best performance by an actress in a television series - drama'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    females_only = only_female_entities(merged)
    show_freq_hosts(females_only)
    name = remove_one_word_names(entities=females_only)
    return name


# best performance by an actor in a television series - drama
def award_winner_21(entities, db):
    print("{award}: ".format(award='best performance by an actor in a television series - drama'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    males_only = only_male_entities(merged)
    show_freq_hosts(males_only)
    name = remove_one_word_names(entities=males_only)
    return name


# 'best performance by an actress in a television series - comedy or musical'
def award_winner_22(entities, db):
    print("{award}: ".format(award='best performance by an actress in a television series - comedy or musical'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    females_only = only_female_entities(merged)
    show_freq_hosts(females_only)
    name = remove_one_word_names(entities=females_only)
    return name


# 'best performance by an actor in a television series - comedy or musical'
def award_winner_23(entities, db):
    print("{award}: ".format(award='best performance by an actor in a television series - comedy or musical'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    males_only = only_male_entities(merged)
    show_freq_hosts(males_only)
    name = remove_one_word_names(entities=males_only)
    return name


# 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television'
def award_winner_24(entities, db):
    print("{award}: ".format(award='best performance by an actress in a supporting role in a series, limited series or motion picture made for television'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    females_only = only_female_entities(merged)
    show_freq_hosts(females_only)
    name = remove_one_word_names(entities=females_only)
    return name


# 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television'
def award_winner_25(entities, db):
    print("{award}: ".format(award='best performance by an actor in a supporting role in a series, limited series or motion picture made for television'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    males_only = only_male_entities(merged)
    show_freq_hosts(males_only)
    name = remove_one_word_names(entities=males_only)
    return name


# 'cecil b. demille award'
def award_winner_26(entities, db):
    print("{award}: ".format(award='cecil b. demille award'))
    shorten = shorten_dict(entities)
    sub = merge_keys_for_person_winner_sub(shorten)
    merged = merge_keys_for_person_winner_imdb(sub, db=db)
    show_freq_hosts(merged)
    return max(merged, key=merged.get)


if __name__ == '__main__':
    main()
