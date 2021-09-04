import pickle
import sys
import time
from ExtractFeatures import extract

# Command-line arguments
input_file_name = sys.argv[1]
model_file_name = sys.argv[2]
feature_map_file = sys.argv[3]
output_file = sys.argv[4]

UNK_TOKEN = 'UNK'
START_TOKEN = '<START>'
START_TAG = '<S>'
END_TOKEN = '<END>'


if __name__ == "__main__":

    start_time = time.time()

    # Read the input data.
    with open(input_file_name, "r", encoding="utf-8") as input_file:
        input_data = input_file.readlines()

    # Parse the sentences in the input data and find the longest sentence length.
    all_sent = [line.strip().split(' ') for line in input_data]
    len_longest_sent = max([len(sent) for sent in all_sent])

    # List to hold the last two predictions of each sentence in the data.
    last_two_tags = [(START_TAG, START_TAG) for _ in range(len(all_sent))]

    # Dictionary to hold the predictions of each sentence in the data.
    tags = {new_list: [] for new_list in range(len(all_sent))}

    # Load the DictVectorizer object and the training set words.
    with open(feature_map_file, 'rb') as file:
        v = pickle.load(file)
        train_words = pickle.load(file)

    # Load the trained Logistic Regression model.
    log_reg = pickle.load(open(model_file_name, 'rb'))

    # For word in position i in each sentence.
    for i in range(len_longest_sent):

        # Gather all sentence indexes participating in the current iteration.
        cur_iteration_sentences = [j for j, sent in enumerate(all_sent) if i + 1 <= len(sent)]

        position_i_features_dicts = []

        # For word in position i in each of the current iteration sentences.
        for sent_idx in cur_iteration_sentences:

            # Extract the word's features.
            features = extract(all_sent[sent_idx], i, last_two_tags[sent_idx], not all_sent[sent_idx][i] in train_words)
            position_i_features_dicts.append(features)

        # Convert all features of all words in position i into feature vectors.
        feature_vectors = v.transform(position_i_features_dicts)

        # Predict tag for each words in position i of the current iteration's sentences.
        pred_tags = log_reg.predict(feature_vectors)

        # Save the prediction to each word in position i and update for each sentence its last two tags.
        for sent_idx, tag in zip(cur_iteration_sentences, pred_tags):
            tags[sent_idx].append(tag)
            last_two_tags[sent_idx] = (last_two_tags[sent_idx][1], tag)

    # Write all predictions to the output file.
    with open(output_file, "w") as out_file:
        for sent_idx, line in enumerate(all_sent):
            for i, (word, tag) in enumerate(zip(line, tags[sent_idx])):
                out_file.write('{}/{}{}'.format(word, tag, ' ' if i + 1 < len(line) else '\n'))

    passed_time = time.time() - start_time
    print("Prediction finished in %.2f seconds" % passed_time)
