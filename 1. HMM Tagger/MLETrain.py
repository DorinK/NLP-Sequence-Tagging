import sys
from collections import Counter
import re
import time

# Command-line arguments
input_file_name = sys.argv[1]
q_mle = sys.argv[2]
e_mle = sys.argv[3]

START_TAG = '<S>'
END_TAG = '<E>'


def read_and_analyze_data():

    words, tags, words_tags = [], [], []
    trigrams, bigrams, unigrams = [], [], []

    # Read the input data.
    with open(input_file_name, "r", encoding="utf-8") as input_file:
        input_data = input_file.readlines()

    # For each sentence in the input data.
    for sentence in input_data:

        # Parse the sentence.
        tuples = [tuple(pair.rsplit('/', 1)) for pair in sentence.strip().split(' ')]
        words += [pair[0] for pair in tuples]
        tags += [pair[1] for pair in tuples]
        words_tags += tuples

        # Add the special symbols at the start and at the end of the sentence.
        sent_tags = [START_TAG, START_TAG] + [pair[1] for pair in tuples] + [END_TAG]

        # Catch all trigrams, bigrams and unigrams in the sentence.
        trigrams += list(zip(sent_tags, sent_tags[1:], sent_tags[2:]))
        bigrams += list(zip(sent_tags, sent_tags[1:]))
        unigrams += sent_tags

    return words, tags, words_tags, trigrams, bigrams, unigrams


def calcQ(trigrams, bigrams, unigrams):

    # The counts of all the trigrams, bigrams and unigrams in the training set.
    q_counts = Counter(unigrams + [' '.join(s) for s in bigrams] + [' '.join(s) for s in trigrams])

    # Produce q.mle file with the q counts.
    with open(q_mle, "w") as q_file:
        for count in q_counts:
            q_file.write("%s\t%d\n" % (count, q_counts[count]))


def count_patterns(words_tags, words_tags_counter, num_occurrences=1, unknown_token='UNK'):

    # Count the number of occurrences of each word in the training set.
    counter = Counter([pair[0] for pair in words_tags])

    rare = set()

    # Collect the words in the training set that appear only once and consider them as rare wards.
    for word, amount in counter.items():
        if amount <= num_occurrences:
            rare.add(word)

    pattern_e_counts = Counter()

    # Go over each word and its associated tag that were found in the training set.
    for word, tag in set(words_tags):

        if word in rare:    # If the word is rare.

            # Find its fit word-signature pattern and update the word-signature count for its associated tag.
            signature = word_signature(word) if word_signature(word) is not None else unknown_token
            pattern_e_counts[signature + ' ' + tag] += words_tags_counter[word + ' ' + tag]

        else:   # Otherwise

            # Find the fit word-signature pattern, if exists.
            signature = word_signature(word)

            # If there is a word-signature pattern that fits.
            if signature is not None:

                # Then update the word-signature count for the word's associated tag.
                pattern_e_counts[signature + ' ' + tag] += words_tags_counter[word + ' ' + tag]

    return pattern_e_counts


def calcE(words_tags):

    # The counts of every word and the tag associated with it in the training set.
    e_counts = Counter([' '.join(s) for s in words_tags])

    # Compute the counts of word signatures with the relevant tags.
    pattern_e_counts = count_patterns(words_tags, e_counts)

    # Produce e.mle file with the e counts.
    with open(e_mle, "w") as e_file:

        for count in e_counts:
            e_file.write("%s\t%d\n" % (count, e_counts[count]))

        for count in pattern_e_counts:
            e_file.write("^%s\t%d\n" % (count, pattern_e_counts[count]))


def word_signature(word):

    # Some general patterns:
    if re.search(r'^[0-9]+[,/.][0-9]+[,]?[0-9]*$', word) is not None:
        return 'UNK_NUM'
    if re.search(r'^[0-9]+:[0-9]+$', word) is not None:
        return 'UNK_HOUR'
    if re.search(r'^[0-9]+/[0-9]+-[a-zA-Z]+[-]?[a-zA-Z]*$', word) is not None:
        return 'UNK_FRUC-WORD'
    if re.search(r'^[A-Z]+-[A-Z]+$', word) is not None:
        return 'UNK_AA-AA'
    if re.search(r'^[a-z]+-[a-z]+$', word) is not None:
        return 'UNK_aa-aa'
    if re.search(r'^[A-Z][a-z]+-[A-Z][a-z]+$', word) is not None:
        return 'UNK_Aa-Aa'
    if re.search(r'^[A-Z]+$', word) is not None:
        return 'UNK_UPPER_CASE'
    if re.search(r'^[A-Z][a-z]+$', word) is not None:
        return 'UNK_Aa'

    if word[-3:] == 'ing':
        return 'UNK_ING'
    if word[-2:] == 'ed':
        return 'UNK_ED'
    if word[-3:] == 'ure':
        return 'UNK_URE'
    if word[-3:] == 'age':
        return 'UNK_AGE'

    # Noun Suffixes:
    if word[-3:] == 'acy':
        return 'UNK_ACY'
    if word[-2:] == 'al':
        return 'UNK_AL'
    if word[-4:] == 'ance' or word[-4:] == 'ence':
        return 'UNK_ANCE'
    if word[-3:] == 'dom':
        return 'UNK_DOM'
    if word[-2:] == 'er' or word[-2:] == 'or':
        return 'UNK_ER'
    if word[-3:] == 'ism':
        return 'UNK_ISM'
    if word[-3:] == 'ist':
        return 'UNK_IST'
    if word[-2:] == 'ty' or word[-3:] == 'ity':
        return 'UNK_TY'
    if word[-4:] == 'ment':
        return 'UNK_MENT'
    if word[-4:] == 'ness':
        return 'UNK_NESS'
    if word[-4:] == 'ship':
        return 'UNK_SHIP'
    if word[-4:] == 'tion':
        return 'UNK_TION'
    if word[-4:] == 'sion':
        return 'UNK_SION'

    # Verb Suffixes:
    if word[-3:] == 'ate':
        return 'UNK_ATE'
    if word[-2:] == 'en':
        return 'UNK_EN'
    if word[-2:] == 'fy':
        return 'UNK_FY'
    if word[-3:] == 'ify':
        return 'UNK_IFY'
    if word[-3:] == 'ize' or word[-3:] == 'ise':
        return 'UNK_IZE'

    # Adjective Suffixes:
    if word[-4:] == 'able' or word[-4:] == 'ible':
        return 'UNK_ABLE'
    if word[-2:] == 'al':
        return 'UNK_AL'
    if word[-3:] == 'ful':
        return 'UNK_FUL'
    if word[-2:] == 'ic' or word[-4:] == 'ical':
        return 'UNK_IC'
    if word[-4:] == 'ious' or word[-3:] == 'ous':
        return 'UNK_OUS'
    if word[-3:] == 'ish':
        return 'UNK_ISH'
    if word[-3:] == 'ive':
        return 'UNK_IVE'
    if word[-4:] == 'less':
        return 'UNK_LESS'

    return None


if __name__ == '__main__':

    start_time = time.time()

    # Parse the input data.
    words, tags, words_tags, trigrams, bigrams, unigrams = read_and_analyze_data()

    # Compute the e and q counts and write them to .mle and q.mle files respectively.
    calcE(words_tags)
    calcQ(trigrams, bigrams, unigrams)

    passed_time = time.time() - start_time
    print("Training finished in %.2f seconds" % passed_time)
