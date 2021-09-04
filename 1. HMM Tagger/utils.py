import re
from collections import defaultdict

START_TAG = '<S>'
END_TAG = '<E>'


def read_q_mle_file(q_mle):

    # Extract the q counts from the q.mle file.
    with open(q_mle, "r", encoding="utf-8") as q_file:
        counts = q_file.readlines()

    q_counts = {}
    total_tags = 0

    # For each count in the q counts.
    for q_count in counts:

        # Parse the line.
        tag_sequence, quantity = q_count.strip().split('\t')
        q_counts[tag_sequence] = int(quantity)

        # Count the number of tags occurrences in the training data.
        if tag_sequence == tag_sequence.strip().split(' ')[0] and tag_sequence != START_TAG and tag_sequence != END_TAG:
            total_tags += int(quantity)

    return q_counts, total_tags


def read_e_mle_file(e_mle):

    # Extract the e counts from the e.mle file.
    with open(e_mle, "r", encoding="utf-8") as e_file:
        counts = e_file.readlines()

    e_counts = {}
    word_possible_tags = defaultdict(set)

    # For each count in the e counts.
    for e_count in counts:

        # Parse the line.
        word_tag, quantity = e_count.strip().split('\t')
        e_counts[word_tag if word_tag[0] is not '^' else word_tag[1:]] = int(quantity)

        word, tag = word_tag.strip().split(' ')
        word_possible_tags[word if word[0] is not '^' else word[1:]].add(tag)

    return e_counts,  word_possible_tags


def find_word_signature(word):

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

    return 'UNK'
