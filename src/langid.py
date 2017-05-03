""" Language Identification System
    Jack Kaloger 2017
    COMP30027 Project 2
"""


import operator
import json
import numpy as np
from sklearn.feature_extraction.text import *


langs = ["ar", "bg", "de", "en", "es", "fa", "fr",
         "he", "hi", "it", "ja", "ko", "mr", "ne",
         "nl", "ru", "th", "uk", "ur", "zh", "unk"]

################################################################################
# parsing/utility functions
################################################################################


def read_json(filename):
    a = []
    for line in open(filename):
        a.append(json.loads(line))
    return a


def get_langs(data):
    return [x["lang"] for x in data]


################################################################################
# n-gram Functions
################################################################################


def get_word_ngrams(text, n):
    a = text.split()
    grams = [a[x:x+n] for x in range(len(a))]
    return grams


def get_byte_ngrams(text, n):
    grams = [text[x:x+n] for x in range(len(text))]
    return grams



################################################################################
# doc-term matrix
################################################################################


def get_feature_vector(v, text):
    return v.transform([text])


def get_document_term_matrix(data):
    v = CountVectorizer()
    corpus = [d["text"] for d in data]
    '''" ".join(filter(lambda x: x[0] != '#', d["text"].split()))'''
    t = v.fit_transform(corpus)
    a = v.build_analyzer()
    return [v, t, a]  # change to dictionary


################################################################################
# Nearest Prototype Classifier
################################################################################


def nc_eval(tl, dd, dl, v):
    centroids = get_centroids(tl, v)
    labels = []
    for instance in dd:
        label = nearest_prototype(centroids, get_feature_vector(v[0], instance["text"]))
        labels.append(label)

    return accuracy(labels, dl)


def nearest_prototype(centroids, instance):
    dists = dict((key, 0) for key in langs)
    for centroid in centroids:
        dists[centroid] = euclid_dist(instance.toarray()[0], centroids[centroid])
    return sorted(dists.items(), key=operator.itemgetter(1))[0][0]


def get_centroids(tl, v):
    centroids = dict((key, 0) for key in langs)
    count = dict((key, 0) for key in langs)
    for i, row in enumerate(v[1].toarray()):
        centroids[tl[i]] += row
        count[tl[i]] += 1
    for key, value in centroids.items():
        centroids[key] = value / count[key]

    return centroids


def euclid_dist(a, b):
    return np.linalg.norm(a-b)


################################################################################
# Naïve Bayes Classifier
################################################################################
################################################################################
# Decision Tree Classifier
################################################################################
################################################################################
# Evaluation Functions
################################################################################


def accuracy(predicted, real):
    t = 0
    n = 0
    for p, r in zip(predicted, real):
        if p == r:  # when we predicted correctly (TP OR TN)
            t += 1
        else:  # (FP OR FN)
            n += 1
    return t/(t + n)  # equiv to (TP+TN)/(TP+TN+FP+FN)

################################################################################
# main()
################################################################################
if __name__ == "__main__":
    training_data = read_json('in/train.json')
    training_labels = get_langs(training_data)

    dev_data = read_json('in/dev.json')
    dev_labels = get_langs(dev_data)

    test_data = read_json('in/test.json')

    V = get_document_term_matrix(training_data)

    print(nc_eval(training_labels, dev_data, dev_labels, V))
