import numpy as np
import os
import csv
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

FIRST_PERSON_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
SECOND_PERSON_PRONOUNS = {
    'you', 'your', 'yours', 'u', 'ur', 'urs'}
THIRD_PERSON_PRONOUNS = {
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them',
    'their', 'theirs'}
SLANG = {
    'smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff',
    'wyd', 'lylc', 'brb', 'atm', 'imao', 'sml', 'btw', 'bw', 'imho', 'fyi',
    'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
    'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya',
    'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'}
PUNCTUATION = [
    '#',
    '$',
    '.',
    ',',
    ':',
    '(',
    ')',
    '"',
    '‘',  # open quote
    '“',  # double open
    '’'  # single close
    '”',  # double close
]

# must define global variable
global bristol_gilhooly
global warriner


def extract_text(comment):
    """
    returns the text of the comment with the punctuation and parts of speech tags removed
    """
    result = []
    flag = False  # deals with case when lemma is split into 2 words, doesn't handle case where lemma is three words though
    for word in comment.split():
        if flag:
            result[-1] = result[-1] + " " + word.split('/')[0]
            flag = False
        else:
            result.append(word.split('/')[0])
        if "/" not in word:
            flag = True

    return result


def extract_tags(comment):
    """
    returns the tags of the comment
    """
    if len(comment) == 0:
        return ""

    result = []
    for word in comment.split():
        if "/" not in word:
            pass
        else:
            result.append(word.split("/")[1])

    return result


def count_uppercase(text):
    """
    takes in cleaned text
    """
    count = 0
    for word in text:
        if word.isupper() and len(word) >= 3:
            count += 1

    return count


def count_contains(text, word_list):
    """
    text is cleaned text (no PoS tags)
    returns how many words from text are also in the given word_list
    """
    count = 0
    for word in text:
        if word in word_list:
            count += 1

    return count


def average_sentence_length(comment, text):
    """
    uncleaned comment, i.e directly from preproc
    average length of sentence in tokens!
    """
    num_sentences = number_of_sentences(comment)
    if num_sentences == 0:
        return 0

    return len(text) / num_sentences


# TODO: how to handle case where lemma is 2 words, should those 2 words be treated as 1 token or as 2 tokens?
# currently treating them as single unit
def average_token_length(text, tags):
    """
    average lenth of token in characters
    """
    # from appendix table 3 in the handout
    if len(text) == 0:
        return 0.00
    total_token_length = 0.00
    total_number_of_tokens = 0
    # assert len(text) == len(tags)
    for word, tag in zip(text, tags):
        if tag not in PUNCTUATION:
            total_token_length += len(word)
            total_number_of_tokens += 1
    if total_number_of_tokens == 0:
        return 0.00
    return total_token_length / total_number_of_tokens


def number_of_sentences(comment):
    sentences = comment.split("\n")
    total = 0
    for sent in sentences:
        if len(sent) != 0:
            total += 1
    return total


def multi_character_punctuation(text):
    result = 0
    for word in text:
        is_punctuation = True
        for ch in word:
            if ch not in PUNCTUATION:
                is_punctuation = False
                break
        if is_punctuation and len(word) > 1:
            result += 1

    return result


def future_tense_verbs(text, tags):
    result = 0
    for index in range(len(text)):
        word = text[index]
        tag = tags[index]
        if word == "will" and tag == "MD":
            result += 1
        try:
            if tag == "VBG" and text[index + 1].lower() == "to" and tags[index + 2] == "VB":
                result += 1
        except:
            # just swallow this error, better to ask for forgiveness than ask for permissions
            pass

    return result


def wordlist_average(wordlist, index, text):
    if len(text) == 0:
        return 0.00

    # case for each word is blank
    empty = True
    for word in text:
        if len(word) != 0:
            empty = False
    if empty:
        return 0.00

    all_words = []
    all_values = []
    for word in text:
        row = wordlist[word]
        if row and row[index] != 'NA' and row[index] != '':  # only count words that appear in the word list
            all_values.append(float(row[index]))
            all_words.append(word)
    if all_values:
        return np.mean(all_values)
    return 0.00


def wordlist_standard_deviation(wordlist, index, text):
    if len(text) == 0:
        return 0.00

    empty = True
    for word in text:
        if len(word) != 0:
            empty = False
    if empty:
        return 0.00

    values = []
    for word in text:
        row = wordlist[word]
        if row and row[index] != 'NA' and row[index] != '':  # only count words that appear in the word list
            values.append(float(row[index]))
    if len(values) <= 1:
        return 0.00
    return np.std(values)


def extract1(comment):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    feats = np.zeros((1, 173))
    text = extract_text(comment)
    tags = extract_tags(comment)
    feats[0, 0] = count_uppercase(text)  # Number of tokens in uppercase (≥ 3 letters long)
    text = [w.lower() for w in text]  # conver to lowercase after extracting only feature that is capitalization sensitive

    feats[0, 1] = count_contains(text, FIRST_PERSON_PRONOUNS)  # Number of first-person pronouns
    feats[0, 2] = count_contains(text, SECOND_PERSON_PRONOUNS)  # Number of second-person pronouns
    feats[0, 3] = count_contains(text, THIRD_PERSON_PRONOUNS)  # Number of third-person pronouns
    feats[0, 4] = count_contains(tags, ["CC"])  # coordnating conjunction
    feats[0, 5] = count_contains(tags, ["VBD"])  # past-tense verb
    feats[0, 6] = future_tense_verbs(text, tags)  # future-tense verb
    feats[0, 7] = count_contains(tags, [","])  # number of commas
    feats[0, 8] = multi_character_punctuation(text)  # multi-character punctuation tokens
    feats[0, 9] = count_contains(tags, ["NN", "NNS"])  # common nouns
    feats[0, 10] = count_contains(tags, ["NNP", "NNPS"])  # proper nouns
    feats[0, 11] = count_contains(tags, ["RB", "RBR", "RBS"])  # adverbs
    feats[0, 12] = count_contains(tags, ["WDT", "WP", "WP$", "WRB"])  # wh- words
    feats[0, 13] = count_contains(text, SLANG)  # slang acronymns
    feats[0, 14] = average_sentence_length(comment, text)  # average length of sentences in tokens
    feats[0, 15] = average_token_length(text, tags)  # average length of tokens, excluding punctuation-only tokens, in characters
    feats[0, 16] = number_of_sentences(comment)  # TODO: fix
    # BGL wordlist
    global bristol_gilhooly
    feats[0, 17] = wordlist_average(bristol_gilhooly, 3, text)  # Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    feats[0, 18] = wordlist_average(bristol_gilhooly, 4, text)  # Average of IMG from Bristol, Gilhooly, and Logie norms
    feats[0, 19] = wordlist_average(bristol_gilhooly, 5, text)  # Average of FAM from Bristol, Gilhooly, and Logie norms
    feats[0, 20] = wordlist_standard_deviation(bristol_gilhooly, 3,
                                               text)  # Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    feats[0, 21] = wordlist_standard_deviation(bristol_gilhooly, 4, text)  # Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    feats[0, 22] = wordlist_standard_deviation(bristol_gilhooly, 5, text)  # Standard deviation of FAM from Bristol, Gilhooly, and Logie norms

    # Warriner
    global warriner
    feats[0, 23] = wordlist_average(warriner, 2, text)  # Average of V.Mean.Sum from Warringer norms
    feats[0, 24] = wordlist_average(warriner, 5, text)  # Average of A.Mean.Sum from Warringer norms
    feats[0, 25] = wordlist_average(warriner, 8, text)  # Average of D.Mean.Sum from Warringer norms
    feats[0, 26] = wordlist_standard_deviation(warriner, 2, text)  # Standard deviation of V.Mean.Sum from Warringer norms
    feats[0, 27] = wordlist_standard_deviation(warriner, 5, text)  # Standard deviation of A.Mean.Sum from Warringer norms
    feats[0, 28] = wordlist_standard_deviation(warriner, 8, text)  # Standard deviation of D.Mean.Sum from Warringer norms
    return feats


def load_wordlist(file):
    dictionary = defaultdict(lambda: [])  # helps when detecting if word exists in the given wordlist
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # this takes the last value from the wordlist, this matches the sample given out
            dictionary[row[1]] = row

    return dictionary


def load_feats(file):
    return np.load(file)


def load_indices(file):
    """
    returns a dictionary in the form of key=id, value=index
    """
    dictionary = {}
    with open(file) as fp:
        index = 0
        line = fp.readline()
        while line:
            dictionary[line.strip()] = index
            line = fp.readline()
            index += 1

    return dictionary


def main():
    # Declare necessary global variables here.
    # load wordlists
    global bristol_gilhooly
    path = os.path.join("Wordlists", "BristolNorms+GilhoolyLogie.csv")
    bristol_gilhooly = load_wordlist(path)

    global warriner
    path = os.path.join("Wordlists", "Ratings_Warriner_et_al.csv")
    warriner = load_wordlist(path)

    preprocessed_data: pd.dataframe
    preprocessed_data = pd.read_csv(os.path.join("data", "processed_data.csv"))

    feats = np.zeros((len(preprocessed_data), 29 + 1))  # 29 features, 1 label

    for index, datum in tqdm(preprocessed_data.iterrows(), total=preprocessed_data.shape[0]):
        feats[index, 0:29] = extract1(datum["Phrase"])[0, 0:29]
        feats[index, 29] = datum["Sentiment"]

    np.savez_compressed(os.path.join("data", "feature_extracted_data"), feats)


# TODO: extensions
# how we treat blank strings, how we treat deleted or removed, that might be something more about how the subreddit mod's work than the text
# reduce dimensionality, unsupervised learning as some form of preprocessing to extract most important features before training

if __name__ == "__main__":
    main()
