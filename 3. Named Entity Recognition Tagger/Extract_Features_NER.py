import pickle
import sys
import time
from collections import Counter, defaultdict

# Command-line arguments
corpus_file = sys.argv[1]
features_file = sys.argv[2]

UNK_TOKEN = 'UNK'
START_TOKEN = '<START>'
START_TAG = '<S>'
END_TOKEN = '<END>'
END_TAG='<E>'

person_lex = set()
loc_lex = set()
org_lex = set()


def extract(sent, i, last_tags, rare_or_unknown,pos_tags):

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
                    next_next_w=_sent[_i + 2],
                    pos_tag=pos_tags[i],
                    is_person=True if word.upper() in person_lex or word.lower() in person_lex or word[:1].upper() + word[1:].lower() in person_lex else False,)

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

    with open(corpus_file, "r", encoding="utf-8") as input_file:
        corpus = input_file.readlines()

    all_words = [pair.rsplit('/', 1)[0] for line in corpus for pair in line.strip().split(' ')]
    rare_words = identify_rare_words(all_words)

    with open('./lex/firstname.5k', "r", encoding="utf-8") as lex:
        person_lex.update(lex.readlines())

    with open('./lex/firstname.1000', "r", encoding="utf-8") as lex:
        person_lex.update(lex.readlines())

    with open('./lex/lastname.5000', "r", encoding="utf-8") as lex:
        person_lex.update(lex.readlines())

    with open('./lex/people.family_name', "r", encoding="utf-8") as lex:
        person_lex.update(lex.readlines())

    with open('./lex/people.person.lastnames', "r", encoding="utf-8") as lex:
        person_lex.update(lex.readlines())

    with open('./lex/location', "r", encoding="utf-8") as lex:
        loc_lex.update(lex.readlines())

    with open('./lex/location.country', "r", encoding="utf-8") as lex:
        loc_lex.update(lex.readlines())

    with open('./lex/venues', "r", encoding="utf-8") as lex:
        loc_lex.update(lex.readlines())

    with open('./lex/venture_capital.venture_funded_company', "r", encoding="utf-8") as lex:
        org_lex.update(lex.readlines())

    with open('./lex/automotive.make', "r", encoding="utf-8") as lex:
        org_lex.update(lex.readlines())

    with open('./lex/business.brand', "r", encoding="utf-8") as lex:
        org_lex.update(lex.readlines())

    with open('./lex/business.sponsor', "r", encoding="utf-8") as lex:
        org_lex.update(lex.readlines())

    with open('./viterbi_out_MHMM_train', "r", encoding="utf-8") as pos_train:
        pos_data=pos_train.readlines()

    word_possible_tags = defaultdict(set)
    # with open(corpus_file, "r") as corpus:
    with open(features_file, "w") as features_file:

        for sentence, pos_seq in zip(corpus, pos_data):

            tuples = [tuple(pair.rsplit('/', 1)) for pair in sentence.strip().split(' ')]
            words, tags = [pair[0] for pair in tuples], [START_TAG, START_TAG] + [pair[1] for pair in tuples]

            pos_tags =[tuple(pair.rsplit('/', 1))[1] for pair in pos_seq.strip().split(' ')]

            # TODO: TO make only the last two tags pass to extract
            for idx in range(len(tuples)):

                two_last_tags = tags[idx], tags[idx + 1]

                features = extract(words, idx, two_last_tags,words[idx] in rare_words,pos_tags)

                features_file.write(
                    tags[idx + 2] + ' ' + ' '.join([key + "=" + str(val) for key, val in features.items()]) + "\n")

                if words[idx] in rare_words:
                    word_possible_tags[UNK_TOKEN].add(tags[idx + 2])
                else:
                    word_possible_tags[words[idx]].add(tags[idx + 2])

                if idx + 1 == len(tuples):
                    word = END_TOKEN
                    sent = words
                    features = dict(prev_prev_t=two_last_tags[1],
                                    prev_t=tags[idx + 2],
                                    word=word,
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
                                    prev_prev_w=sent[idx - 1],
                                    prev_w=sent[idx],
                                    next_w=END_TOKEN,
                                    next_next_w='',
                                    pos_tag='<<end>>',
                                    is_person=False,)
                                    # is_location=False)

                    features_file.write(
                        END_TAG + ' ' + ' '.join([key + "=" + str(val) for key, val in features.items()]) + "\n")
                    word_possible_tags[word].add(END_TAG)

    passed_time = time.time() - start_time
    print("Feature Extraction finished in %.2f seconds" % passed_time)

    # # TODO: Make reading from a file with relevant name
    with open('./extra_file_MHMM.txt', "w") as file:
        for word, tags in word_possible_tags.items():
            file.write("{}\t{}\n".format(word, ' '.join(tags)))

    # TODO: When final should make features_file_partial


     # # TODO: WHEN DONE REMOVE ACCURACY CALC
    # with open(output_file, "r", encoding="utf-8") as out_file:
    #     out = out_file.readlines()
    # all_preds = [pair.rsplit('/', 1)[1] for line in out for pair in line.strip().split(' ')]
    #
    # # with open('./data/ass1-tagger-dev', "r", encoding="utf-8") as input_file:
    # with open('./ner/dev', "r", encoding="utf-8") as input_file:  # FOR NER
    #     input = input_file.readlines()
    # all_gold = [pair.rsplit('/', 1)[1] for line in input for pair in line.strip().split(' ')]
    #
    # correct = 0
    # total = 0  # FOR NER
    #
    # # correct += sum([i == j for i, j in zip(all_gold, all_preds)])
    #
    # for pred, tag in zip(all_preds, all_gold):  # FOR NER
    #     total += 1
    #     # Don't count the cases in which both prediction and tag are 'O'.
    #     if pred == tag:
    #         if pred == 'O':
    #             total -= 1
    #         else:
    #             correct += 1
    #
    # # print("Accuracy of Local Feature-based Tagger: {:.3f}\n".format((correct / len(all_gold)) * 100))
    # print("Accuracy of Local Feature-based Tagger: {:.3f}\n".format((correct / total) * 100))  # FOR NER
