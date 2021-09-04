import itertools
import pickle
import sys
import time
import numpy as np
from collections import defaultdict, deque

input_file_name = sys.argv[1]
model_file_name = sys.argv[2]
feature_map_file = sys.argv[3]
MEMM_output = sys.argv[4]
extra_file = sys.argv[5]

UNK_TOKEN = 'UNK'
START_TOKEN = '<START>'
START_TAG = '<S>'
END_TOKEN = '<END>'
END_TAG = '<E>'  # TODO: NO NEED

word_possible_tags = {}
person_lex = set()
loc_lex=set()
org_lex = set()

# TODO: Make reading from a file with relevant name
def load_extra_file():
    with open(extra_file, "r", encoding="utf-8") as file:
        data = file.readlines()

    for line in data:
        word, possible_tags = line.strip().split('\t')
        word_possible_tags[word] = possible_tags.strip().split(' ')

    word_possible_tags[START_TOKEN] = [START_TAG]


# TODO: Make this file use the function that in ExtractFeatures

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


def extract_end(sentence, last_tags):
    word = END_TOKEN

    features = dict(prev_prev_t=last_tags[0],
                    prev_t=last_tags[1],
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
                    prev_prev_w=sentence[len(sentence) - 2],
                    prev_w=sentence[len(sentence) - 1],
                    next_w=END_TOKEN,
                    next_next_w='',
                    pos_tag='<<end>>',
                    is_person=False,)

    return features


def MHMM_tagger(sentence, log_reg, dict_vec, train_words,pos_tags):

    V = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
    B = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    V[0][START_TAG][START_TAG] = 1
    possible_tags_per_position = [[START_TAG], [START_TAG]]
    classes_clf = log_reg.classes_

    k = 0
    for k in range(1, len(sentence) + 1):

        word = sentence[k - 1] if sentence[k - 1] in train_words else UNK_TOKEN
        # features = extract(sentence, k - 1, [w, u], train_words)
        possible_tags_per_position.append(word_possible_tags[word])

        tag_pairs = list(itertools.product(possible_tags_per_position[k], possible_tags_per_position[k + 1]))

        for u, v in tag_pairs:

            V[k][u][v] = np.max(
                [V[k - 1][w][u] +
                 log_reg.predict_log_proba(dict_vec.transform(extract(sentence, k - 1, [w, u], not sentence[k - 1] in train_words,pos_tags)))[0][
                     int(np.where(log_reg.classes_ == v)[0])] for w in possible_tags_per_position[k - 1]])

            B[k][u][v] = (possible_tags_per_position[k - 1])[int(np.argmax(
                [V[k - 1][w][u] +
                 log_reg.predict_log_proba(dict_vec.transform(extract(sentence, k - 1, [w, u], not sentence[k - 1] in train_words,pos_tags)))[0][
                     int(np.where(log_reg.classes_ == v)[0])] for w in possible_tags_per_position[k - 1]]))]

    tag_pairs = list(itertools.product(possible_tags_per_position[k], possible_tags_per_position[k + 1]))

    y_n_minus_1, y_n = tag_pairs[int(np.argmax(
        [V[k][u][v] + log_reg.predict_log_proba(dict_vec.transform(extract_end(sentence, [u, v])))[0][
            int(np.where(log_reg.classes_ == END_TAG)[0])] for u, v in tag_pairs]))]

    preds = [y_n_minus_1, y_n]
    for k in range(len(sentence) - 2, 0, -1):
        preds.insert(0, B[k + 2][preds[0]][preds[1]])

    return preds if len(sentence) > 1 else preds[1:]


if __name__ == "__main__":

    start_time = time.time()

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

    load_extra_file()

    with open(feature_map_file, 'rb') as file:
        dict_vectorizer = pickle.load(file)
        train_words = pickle.load(file)

    log_reg = pickle.load(open(model_file_name, 'rb'))

    # TODO: Make reading from a file with relevant name
    with open(input_file_name, "r", encoding="utf-8") as file:
        dev_data = file.readlines()

    with open('./viterbi_out_MHMM_test_blind', "r", encoding="utf-8") as pos_test:
        pos_data=pos_test.readlines()

    correct = 0.0
    total = 0.0

    with open(MEMM_output, "w") as file:
        # for line in dev_data:
        for line, pos_seq in zip(dev_data, pos_data):

            pos_tags =[tuple(pair.rsplit('/', 1))[1] for pair in pos_seq.strip().split(' ')]

            # tuples = [tuple(pair.rsplit('/', 1)) for pair in line.strip().split(' ')]
            # words_sequence = [pair[0] for pair in tuples]
            words_sequence = line.strip().split(' ')
            # gold_tags = [pair[1] for pair in tuples]
            pred_tags = MHMM_tagger(words_sequence, log_reg, dict_vectorizer, train_words,pos_tags)
            # correct += sum([i == j for i, j in zip(gold_tags, pred_tags)])
            # total += len(gold_tags)
            # For each prediction and tag of an example in the batch
            # for pred, tag in zip(pred_tags, gold_tags):
            #     total += 1
            #     # Don't count the cases in which both prediction and tag are 'O'.
            #     if pred == tag:
            #         if pred == 'O':
            #             total -= 1
            #         else:
            #             correct += 1
            # copy = ' '.join([pair[0] + "/" + str(pair[1]) for pair in zip(words_sequence, pred_tags)])
            file.write(' '.join([pair[0] + "/" + str(pair[1]) for pair in zip(words_sequence, pred_tags)]) + "\n")

    # print("Accuracy of MHMM Tagger: {:.3f}\n".format((correct / total) * 100))

    passed_time = time.time() - start_time
    print("Prediction finished in %.2f seconds" % passed_time)
