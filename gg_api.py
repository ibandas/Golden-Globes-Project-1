'''Version 0.35'''
import nltk
from nltk.corpus import stopwords, names
from nltk.tree import Tree
import json
from collections import defaultdict
from datetime import datetime, timedelta
import re

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
OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy',
                        'best performance by an actress in a motion picture - drama',
                        'best performance by an actor in a motion picture - drama',
                        'best performance by an actress in a motion picture - musical or comedy',
                        'best performance by an actor in a motion picture - musical or comedy',
                        'best performance by an actress in a supporting role in any motion picture',
                        'best performance by an actor in a supporting role in any motion picture',
                        'best director - motion picture', 'best screenplay - motion picture',
                        'best motion picture - animated', 'best motion picture - foreign language',
                        'best original score - motion picture', 'best original song - motion picture',
                        'best television series - drama', 'best television series - musical or comedy',
                        'best television limited series or motion picture made for television',
                        'best performance by an actress in a limited series or a motion picture made for television',
                        'best performance by an actor in a limited series or a motion picture made for television',
                        'best performance by an actress in a television series - drama',
                        'best performance by an actor in a television series - drama',
                        'best performance by an actress in a television series - musical or comedy',
                        'best performance by an actor in a television series - musical or comedy',
                        'best performance by an actress in a supporting role in a series, limited series or motion picture made for television',
                        'best performance by an actor in a supporting role in a series, limited series or motion picture made for television',
                        'cecil b. demille award']

# Checks/Incorrects for 2013
awards_regex = {
    # Incorrect (Argo wasn't even in the list. It looks like it was considering TV still)
    'best motion picture - drama': ['best', 'motion', 'picture', 'drama'],
    # Incorrect (It came second, don't consider people names for this and it will win)
    'best motion picture - musical or comedy': ['best', 'motion', 'picture', 'musical', 'comedy'],
    # Incorrect (Came fourth after GoldenGlobes, Jennifer Lawrence, and Julianne Moore)
    'best performance by an actress in a motion picture - drama': ['best', 'performance', 'actress', 'motion', 'picture'],
    'best performance by an actor in a motion picture - drama': ['best', 'performance', 'actor', 'motion', 'picture', 'drama'],
    'best performance by an actress in a motion picture - musical or comedy': ['best', 'performance', 'actress', 'motion', 'picture', 'musical', 'comedy'],
    # Half-Check (Only has Jackman and not Hugh Jackman)
    'best performance by an actor in a motion picture - musical or comedy': ['best', 'performance', 'actor', 'motion', 'picture', 'musical', 'comedy'],
    # CHECK (Anne Hathaway)
    'best performance by an actress in a supporting role in any motion picture': ['best', 'performance', 'actress', 'supporting', 'role', 'motion', 'picture'],
    # CHECK
    'best performance by an actor in a supporting role in any motion picture': ['best', 'performance', 'actor', 'supporting', 'role', 'motion', 'picture'],
    # CHECK (Once golden is taken out)
    'best director - motion picture': ['best', 'director', 'motion', 'picture'],
    # CHECK
    'best screenplay - motion picture': ['best', 'screenplay', 'motion', 'picture'],
    # CHECK
    'best motion picture - animated': ['best', 'motion', 'picture', 'animated'],
    # Incorrect (the people came up, but not the movie 'Amour' itself)
    'best motion picture - foreign language': ['best', 'motion', 'picture', 'foreign', 'language'],
    # CHECK (but not by much, will be better once only considers name)
    'best original score - motion picture': ['best', 'original', 'score', 'motion', 'picture'],
    # CHECK (Skyfall - once names and golden globes are taken out)
    'best original song - motion picture': ['best', 'original', 'song', 'motion', 'picture'],
    # CHECK (ONCE GOLDENGLOBES IS TAKEN OUT)
    'best television series - drama': ['best', 'television', 'tv', 'series', 'drama'],
    # CHECK (Once golden globes and people are taken out)
    'best television series - musical or comedy': ['best', 'television', 'tv', 'series', 'musical', 'comedy'],
    'best television limited series or motion picture made for television': ['best', 'television', 'tv', 'limited', 'series', 'motion', 'picture', 'made'],
    'best performance by an actress in a limited series or a motion picture made for television': ['best', 'performance', 'actress', 'limited', 'series', 'motion', 'picture', 'made', 'television', 'tv'],
    'best performance by an actor in a limited series or a motion picture made for television': ['best', 'performance', 'actor', 'limited', 'series', 'motion', 'picture', 'made', 'television', 'tv'],
    'best performance by an actress in a television series - drama': ['best', 'performance', 'actress', 'television', 'tv', 'series', 'drama'],
    # CHECK (Once only names are considered)
    'best performance by an actor in a television series - drama': ['best', 'performance', 'actor', 'television', 'series', 'drama'],
    # Incorrect (Don Cheadle is a close second after only names are considered)
    'best performance by an actress in a television series - musical or comedy': ['best', 'performance', 'actress', 'television', 'tv', 'series', 'musical', 'comedy'],
    'best performance by an actor in a television series - musical or comedy': ['best', 'performance', 'actor', 'television', 'tv', 'series'],
    'best performance by an actress in a supporting role in a series, limited series or motion picture made for television': ['best', 'performance', 'actress', 'supporting', 'role', 'series', 'limited', 'motion', 'picture', 'television', 'tv'],
    # Kind of Check (it's a tie, very low frequency)
    'best performance by an actor in a supporting role in a series, limited series or motion picture made for television': ['best', 'performance', 'actor', 'supporting', 'role', 'series', 'limited', 'motion', 'picture', 'television', 'tv'],
    # Incorrect (Even after Golden Globes is taken out, Jodie Foster is second to Adele by a lot)
    'cecil b. demille award': ['cecil', 'demille', 'award']
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
    result = [i.encode('ascii', 'ignore').strip() for i in continuous_chunk]
    return result


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
                            named_entities[ne] += 1
                        break

    print("\n\n")
    merged = merge_keys(named_entities)
    # show_freq_hosts(merged)
    hosts = calculate_hosts(merged)
    return hosts


# This function is to print out the most common named entities
def show_freq_hosts(ne):
    host_list = []
    freq_count = 0
    freq = nltk.FreqDist(ne)
    for key, val in freq.most_common():
        host_list.append(key)
        freq_count += 1
        print(str(key) + ': ' + str(val))
        if freq_count == 5:
            break
    print('\n')
    return host_list


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
                            del freq[k]
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

    print(hosts)
    return hosts


def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    return []


def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''

    # Go through each tweet. For each tweet look for each award.
    # Build a dictionary for each award.
    # Frequency map similar to hosts for each award.
    # Then take the most frequent named entities from each award and
    # put it in a list as a value for they key of the award

    # Your code here
    return []


def get_winner(year):
    # Create a constant dict of each award as the key
    # and the values are a list of regex for that award

    # Then create a dynamic dict that has each award as the key again
    # but the values are frequencyMap (similar to hosts).
    # So it's a dict nested in a dict


    awards_mapped_to_entities = {
        'best motion picture - drama': defaultdict(int),
        'best motion picture - musical or comedy': defaultdict(int),
        'best performance by an actress in a motion picture - drama': defaultdict(int),
        'best performance by an actor in a motion picture - drama': defaultdict(int),
        'best performance by an actress in a motion picture - musical or comedy': defaultdict(int),
        'best performance by an actor in a motion picture - musical or comedy': defaultdict(int),
        'best performance by an actress in a supporting role in any motion picture': defaultdict(int),
        'best performance by an actor in a supporting role in any motion picture': defaultdict(int),
        'best director - motion picture': defaultdict(int),
        'best screenplay - motion picture': defaultdict(int),
        'best motion picture - animated': defaultdict(int),
        'best motion picture - foreign language': defaultdict(int),
        'best original score - motion picture': defaultdict(int),
        'best original song - motion picture': defaultdict(int),
        'best television series - drama': defaultdict(int),
        'best television series - musical or comedy': defaultdict(int),
        'best television limited series or motion picture made for television': defaultdict(int),
        'best performance by an actress in a limited series or a motion picture made for television': defaultdict(int),
        'best performance by an actor in a limited series or a motion picture made for television': defaultdict(int),
        'best performance by an actress in a television series - drama': defaultdict(int),
        'best performance by an actor in a television series - drama': defaultdict(int),
        'best performance by an actress in a television series - musical or comedy': defaultdict(int),
        'best performance by an actor in a television series - musical or comedy': defaultdict(int),
        'best performance by an actress in a supporting role in a series, limited series or motion picture made for television': defaultdict(int),
        'best performance by an actor in a supporting role in a series, limited series or motion picture made for television': defaultdict(int),
        'cecil b. demille award': defaultdict(int)
    }

    with open('./data/gg{year}.json'.format(year=year)) as f:
        tweets = json.load(f)
        # tweets = [json.loads(line) for line in f]

        start = datetime.now()
        pot = timedelta(seconds=300)
        for tweet in tweets:
            if start + pot < datetime.now():
                break
            tweet_text = tweet.get('text')
            for k in awards_regex:
                rx_counter = 0
                for rx in awards_regex.get(k):
                    if rx in tweet_text:
                        rx_counter += 1
                if rx_counter >= (len(awards_regex.get(k)) / 2):
                    tweet_named_entities = get_continous_chunks(tweet_text)
                    for ne in tweet_named_entities:
                        print(ne)
                        awards_mapped_to_entities[k][ne] += 1
                    break

    print("\n\n")
    # print(awards_mapped_to_entities)
    for award, freq in awards_mapped_to_entities.items():
        print("{award}: ".format(award=award))
        show_freq_hosts(freq)



    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    return []


def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    # Your code here
    return []


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
    return


if __name__ == '__main__':
    main()
