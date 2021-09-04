import itertools
import sys
import time
import numpy as np
from collections import defaultdict
from utils import *

# Command-line arguments
input_file_name = sys.argv[1]
q_mle = sys.argv[2]
e_mle = sys.argv[3]
viterbi_hmm_output = sys.argv[4]
extra_file = sys.argv[5]

START_TAG = '<S>'
END_TAG = '<E>'


class ViterbiHMM:

    def __init__(self):

        self.q_counts, self.total_tags = read_q_mle_file(q_mle)
        self.e_counts,self.word_possible_tags = read_e_mle_file(e_mle)
        self.words_set = set(self.word_possible_tags.keys())

    def viterbi_tagger(self, sentence):

        V = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
        B = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        # The initial probability.
        V[0][START_TAG][START_TAG] = 1
        possible_tags_per_position = [[START_TAG], [START_TAG]]

        k = 0
        for k in range(1, len(sentence) + 1):

            # If the word is unknown, find its fit word signature.
            word = sentence[k - 1] if sentence[k - 1] in self.words_set else find_word_signature(sentence[k - 1])

            # Extract the possible tags for the current word.
            possible_tags_per_position.append(list(self.word_possible_tags[word]))

            # All pairs of possible tags in position k-1 with possible tags in position k.
            tag_pairs = list(itertools.product(possible_tags_per_position[k], possible_tags_per_position[k + 1]))

            # For each pair of possible tag in position k-1 with possible tag in position k.
            for u, v in tag_pairs:

                V[k][u][v] = np.max([V[k - 1][w][u] + np.log(self.getQ(w, u, v)) + np.log(self.getE(word, v)) for w in
                                     possible_tags_per_position[k - 1]])

                B[k][u][v] = (possible_tags_per_position[k - 1])[int(np.argmax(
                    [V[k - 1][w][u] + np.log(self.getQ(w, u, v)) + np.log(self.getE(word, v)) for w in
                     possible_tags_per_position[k - 1]]))]

        # All pairs of possible tags in position k-1 with possible tags in position k.
        tag_pairs = list(itertools.product(possible_tags_per_position[k], possible_tags_per_position[k + 1]))

        # Predict the last two tags of the sentence.
        y_n_minus_1, y_n = tag_pairs[int(np.argmax([V[k][u][v] + np.log(self.getQ(u, v, END_TAG)) for u, v in tag_pairs]))]

        preds = [y_n_minus_1, y_n]

        # Predict the rest of the tags by backtracking.
        for k in range(len(sentence) - 2, 0, -1):
            preds.insert(0, B[k + 2][preds[0]][preds[1]])

        return preds if len(sentence) > 1 else preds[1:]

    def getQ(self, t1, t2, t3):

        trigram = "{} {} {}".format(t1, t2, t3)
        bigram = "{} {}".format(t2, t3)
        unigram = "{}".format(t3)

        # The lambda values for the interpolation.
        lamda3, lamda2, lamda1 = 0.83, 0.09, 0.08

        # Perform a weighted linear interpolation in order to compute the estimate for q.
        q3 = self.q_counts[trigram] / self.q_counts["{} {}".format(t1, t2)] if trigram in self.q_counts.keys() else 0
        q2 = self.q_counts[bigram] / self.q_counts[t2] if bigram in self.q_counts.keys() else 0
        q1 = self.q_counts[unigram] / self.total_tags

        return (lamda3 * q3) + (lamda2 * q2) + (lamda1 * q1)

    def getE(self, wi, ti):

        wi_ti = "{} {}".format(wi, ti)
        return self.e_counts[wi_ti] / self.q_counts[ti]


model = ViterbiHMM()
getQ = model.getQ
getE = model.getE

if __name__ == "__main__":

    start_time = time.time()

    # Read the input data.
    with open(input_file_name, "r", encoding="utf-8") as input_file:
        input_data = input_file.readlines()

    with open(viterbi_hmm_output, "w") as preds_file:

        for sentence in input_data:

            # Parse the sentence.
            words_sequence = sentence.strip().split(' ')

            # Predict the most probable tags sequence for the current sentence.
            pred_tags = model.viterbi_tagger(words_sequence)

            # Write the predictions to the file.
            preds_file.write(' '.join([pair[0] + "/" + str(pair[1]) for pair in zip(words_sequence, pred_tags)]) + "\n")

    passed_time = time.time() - start_time
    print("Prediction finished in %.2f seconds" % passed_time)
