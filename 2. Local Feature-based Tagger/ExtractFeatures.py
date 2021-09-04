import sys
from collections import Counter
import time

# Command-line arguments
corpus_file = sys.argv[1]
features_file = sys.argv[2]

UNK_TOKEN = 'UNK'
START_TOKEN = '<START>'
START_TAG = '<S>'
END_TOKEN = '<END>'


def extract(sent, i, last_tags, rare_or_unknown):

    _sent = [START_TOKEN, START_TOKEN] + sent + [END_TOKEN, END_TOKEN]
    _i = i + 2

    word = _sent[_i]

    features = dict(prev_prev_t=last_tags[0],
                    prev_t=last_tags[1],
                    word=word if not rare_or_unknown else UNK_TOKEN,
                    pref1=word[:1],
                    pref2='' if len(word) < 2 else word[:2],
                    pref3='' if len(word) < 3 else word[:3],
                    pref4='' if len(word) < 4 else word[:4],
                    pref5='' if len(word) < 5 else word[:5],
                    pref6='' if len(word) < 6 else word[:6],
                    suff6='' if len(word) < 6 else word[-6:],
                    suff5='' if len(word) < 5 else word[-5:],
                    suff4='' if len(word) < 4 else word[-4:],
                    suff3='' if len(word) < 3 else word[-3:],
                    suff2='' if len(word) < 2 else word[-2:],
                    suff1=word[-1:],
                    prev_prev_w=_sent[_i - 2],
                    prev_w=_sent[_i - 1],
                    next_w=_sent[_i + 1],
                    next_next_w=_sent[_i + 2])

    return features


def identify_rare_words(words, num_occurrences=1):

    rare = set()

    # Count the number of occurrences of each word in the training set.
    counter = Counter(words)

    # Collect the words in the training set that appear only once and consider them as rare wards.
    for word, amount in counter.items():
        if amount <= num_occurrences:
            rare.add(word)

    return rare


if __name__ == "__main__":

    start_time = time.time()

    # Read the input data.
    with open(corpus_file, "r", encoding="utf-8") as input_file:
        corpus = input_file.readlines()

    # Parse the input sentences and identify the rare words.
    train_words = [pair.rsplit('/', 1)[0] for line in corpus for pair in line.strip().split(' ')]
    rare_words = identify_rare_words(train_words)

    with open(features_file, "w") as features_file:

        for sentence in corpus:

            # Parse the current sentence to words and tags.
            tuples = [tuple(pair.rsplit('/', 1)) for pair in sentence.strip().split(' ')]
            words, tags = [pair[0] for pair in tuples], [START_TAG, START_TAG] + [pair[1] for pair in tuples]

            # For each word in the sentence.
            for idx in range(len(tuples)):

                # Update the last two predicted tags.
                last_two_tags = tags[idx], tags[idx + 1]

                # Extract the word's features.
                features = extract(words, idx, last_two_tags, words[idx] in rare_words)

                # Write the word's features to the features file.
                features_file.write(
                    tags[idx + 2] + ' ' + ' '.join([key + "=" + str(val) for key, val in features.items()]) + "\n")

    passed_time = time.time() - start_time
    print("Feature Extraction finished in %.2f seconds" % passed_time)
