import sys
import time
import numpy as np
from utils import *

# Command-line arguments
input_file_name = sys.argv[1]
q_mle = sys.argv[2]
e_mle = sys.argv[3]
greedy_hmm_output = sys.argv[4]
extra_file = sys.argv[5]

START_TAG = '<S>'
END_TAG = '<E>'


class GreedyHMM:

    def __init__(self):

        self.q_counts, self.total_tags = read_q_mle_file(q_mle)
        self.e_counts, self.word_possible_tags = read_e_mle_file(e_mle)
        self.words_set = set(self.word_possible_tags.keys())

    def greedy_tagger(self, sentence):

        preds = [START_TAG, START_TAG]

        # For each word in the sentence.
        for i in range(len(sentence)):

            # If the word is unknown, find its fit word signature.
            word = sentence[i] if sentence[i] in self.words_set else find_word_signature(sentence[i])

            p_ti_wi = -float("inf")
            tag_i = None

            # For each tag in the list of possible tags of the word.
            for tag in list(self.word_possible_tags[word]):

                # Compute its e and q estimates.
                t_estimation = self.getQ(preds[i], preds[i + 1], tag)
                e_estimation = self.getE(word, tag)

                # Compute the tag's score.
                G_t_wi = np.log(e_estimation) + np.log(t_estimation)

                # If score is better than best score.
                if G_t_wi > p_ti_wi:
                    tag_i = tag
                    p_ti_wi = G_t_wi

            # The tag with the highest score in the current iteration is the prediction for word i.
            preds.append(tag_i)

        return preds[2:]

    def getQ(self, t1, t2, t3):

        trigram = "{} {} {}".format(t1, t2, t3)
        bigram = "{} {}".format(t2, t3)
        unigram = "{}".format(t3)

        # The lambda values for the interpolation.
        lamda3, lamda2, lamda1 = 0.65, 0.21, 0.14

        # Perform a weighted linear interpolation in order to compute the estimate for q.
        q3 = self.q_counts[trigram] / self.q_counts["{} {}".format(t1, t2)] if trigram in self.q_counts.keys() else 0
        q2 = self.q_counts[bigram] / self.q_counts[t2] if bigram in self.q_counts.keys() else 0
        q1 = self.q_counts[unigram] / self.total_tags

        return (lamda3 * q3) + (lamda2 * q2) + (lamda1 * q1)

    def getE(self, wi, ti):

        wi_ti = "{} {}".format(wi, ti)
        return self.e_counts[wi_ti] / self.q_counts[ti]


model = GreedyHMM()
getQ = model.getQ
getE = model.getE

if __name__ == "__main__":

    start_time = time.time()

    # Read the input data.
    with open(input_file_name, "r", encoding="utf-8") as input_file:
        input_data = input_file.readlines()

    with open(greedy_hmm_output, "w") as preds_file:

        for sentence in input_data:

            # Parse the sentence.
            words_sequence = sentence.strip().split(' ')

            # Predict the most probable tags sequence for the current sentence.
            pred_tags = model.greedy_tagger(words_sequence)

            # Write the predictions to the file.
            preds_file.write(' '.join([pair[0] + "/" + str(pair[1]) for pair in zip(words_sequence, pred_tags)]) + "\n")

    passed_time = time.time() - start_time
    print("Prediction finished in %.2f seconds" % passed_time)
