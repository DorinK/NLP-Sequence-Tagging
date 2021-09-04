import pickle
import sys
import time

# Command-line arguments
input_file_name = sys.argv[1]
model_file_name = sys.argv[2]
feature_map_file = sys.argv[3]
output_file = sys.argv[4]

UNK_TOKEN = 'UNK'
START_TOKEN = '<START>'
START_TAG = '<S>'
END_TOKEN = '<END>'

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
                    pref7='' if len(word) < 7 else word[:7],
                    suff7='' if len(word) < 7 else word[-7:],
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
                    is_person=True if word.upper() in person_lex or word.lower() in person_lex or word[:1].upper() + word[1:].lower() in person_lex else False,
                    is_location=True if word.upper() in loc_lex or word.lower() in loc_lex or word[:1].upper() + word[1:].lower() in loc_lex else False,)

    return features


# TODO: Make this more more efficient       V
# TODO: I Think it should only work for untagged data
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

    with open('./viterbi_out_MHMM_dev', "r", encoding="utf-8") as pos_dev:
        pos_data=pos_dev.readlines()


    # Read the input data.
    with open(input_file_name, "r", encoding="utf-8") as input_file:
        input_data = input_file.readlines()

    all_sent = [[tuple(pair.rsplit('/', 1))[0] for pair in line.strip().split(' ')] for line in input_data ]# FOR NER
    # all_sent = [line.strip().split(' ') for line in input_data]
    longest_sent = max([len(sent) for sent in all_sent])

    # List that will hold the last two predictions oof each sentence in the data.
    two_last_tags = [(START_TAG, START_TAG) for _ in range(len(all_sent))]

    # Dictionary to keep the predictions of each sentence in the data.
    tags = {new_list: [] for new_list in range(len(all_sent))}

    # Load the DictVectorizer object and the training set words.
    with open(feature_map_file, 'rb') as file:
        v = pickle.load(file)
        train_words = pickle.load(file)

    # Load the trained logistic regression model.
    log_reg = pickle.load(open(model_file_name, 'rb'))

    # For word in position i in each sentence
    for i in range(longest_sent):

        # Gather all sentence indexes participating in the current iteration.
        cur_iteration_sentences = [j for j, sent in enumerate(all_sent) if i + 1 <= len(sent)]

        position_i_features_dicts = []

        # For word in position i in each of the current iteration's sentences.
        for sent_idx in cur_iteration_sentences:

            pos_tags = [tuple(pair.rsplit('/', 1))[1] for pair in pos_data[sent_idx].strip().split(' ')]

            # Extract features.
            features = extract(all_sent[sent_idx], i, two_last_tags[sent_idx], not all_sent[sent_idx][i] in train_words,pos_tags)
            position_i_features_dicts.append(features)

        # Convert all features into vector features.
        feature_vectors = v.transform(position_i_features_dicts)

        # Predict tag for all words in position i of the current iteration's sentences.
        pred_tags = log_reg.predict(feature_vectors)

        # Save the predictions and update for each sentence its 2 last tags.
        for sent_idx, tag in zip(cur_iteration_sentences, pred_tags):
            tags[sent_idx].append(tag)
            two_last_tags[sent_idx] = (two_last_tags[sent_idx][1], tag)

    # Write all predictions to the output file.
    with open(output_file, "w") as out_file:
        for sent_idx, line in enumerate(all_sent):
            for i, (word, tag) in enumerate(zip(line, tags[sent_idx])):
                out_file.write('{}/{}{}'.format(word, tag, ' ' if i + 1 < len(line) else '\n'))

    passed_time = time.time() - start_time
    print("Prediction finished in %.2f seconds" % passed_time)

     # TODO: WHEN DONE REMOVE ACCURACY CALC
    with open(output_file, "r", encoding="utf-8") as out_file:
        out = out_file.readlines()
    all_preds = [pair.rsplit('/', 1)[1] for line in out for pair in line.strip().split(' ')]

    # with open('./data/ass1-tagger-dev', "r", encoding="utf-8") as input_file:
    with open('./ner/dev', "r", encoding="utf-8") as input_file:  # FOR NER
        input = input_file.readlines()
    all_gold = [pair.rsplit('/', 1)[1] for line in input for pair in line.strip().split(' ')]

    correct = 0
    total = 0  # FOR NER

    # correct += sum([i == j for i, j in zip(all_gold, all_preds)])

    for pred, tag in zip(all_preds, all_gold):  # FOR NER
        total += 1
        # Don't count the cases in which both prediction and tag are 'O'.
        if pred == tag:
            if pred == 'O':
                total -= 1
            else:
                correct += 1

    # print("Accuracy of Local Feature-based Tagger: {:.3f}\n".format((correct / len(all_gold)) * 100))
    print("Accuracy of Local Feature-based Tagger: {:.3f}\n".format((correct / total) * 100))  # FOR NER
