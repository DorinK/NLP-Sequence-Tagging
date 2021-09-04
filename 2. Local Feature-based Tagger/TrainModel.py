import pickle
import sys
import time
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# Command-line arguments
features_file = sys.argv[1]
model_file = sys.argv[2]

words_set = set()


def load_features_file():

    words_features, tags = [], []

    # Fetch the training set's words features.
    with open(features_file, "r", encoding="utf-8") as featuresFile:
        features = featuresFile.readlines()

    # For each line in the features file.
    for line in features:

        # Parse the line into the word's features.
        features = line.strip().split(' ')
        tags.append(features.pop(0))

        # Save the word's features in a dictionary.
        featuresDict = dict([tuple(pair.split('=', 1)) for pair in features])
        words_set.add(featuresDict['word'])
        words_features.append(featuresDict)

    return words_features, tags


def convert_features_to_vectors(features):

    v = DictVectorizer()

    # Convert all words features into feature vectors.
    feature_vectors = v.fit_transform(features)

    # Save the DictVectorizer object to a file for using later in the prediction.
    save_to_file(v)

    return feature_vectors


def save_to_file(feature_map):

    # Save the DictVectorizer object and the set of  words appeared on the training set to a file.
    with open('feature_map_file', 'wb') as file:
        pickle.dump(feature_map, file)
        pickle.dump(words_set, file)


if __name__ == "__main__":

    start_time = time.time()

    # Load all features from the features file and convert them into feature vectors.
    features_per_line, tags = load_features_file()
    feature_vectors = convert_features_to_vectors(features_per_line)

    # Logistic Regression classifier.
    log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial', penalty='l2', tol=1e-4, random_state=0,
                                 max_iter=1200)

    # Train the classifier.
    log_reg.fit(feature_vectors, tags)

    # Save the trained model to model_file.
    pickle.dump(log_reg, open(model_file, 'wb'))

    passed_time = time.time() - start_time
    print("Training finished in %.2f minutes" % (passed_time / 60))
