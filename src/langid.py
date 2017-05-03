""" Language Identification System
    Jack Kaloger 2017
    COMP30027 Project 2
"""

import operator
import json
from sklearn.feature_extraction.text import *


langs = ["ar", "bg", "de", "en", "es", "fa", "fr",
         "he", "hi", "it", "ja", "ko", "mr", "ne",
         "nl", "ru", "th", "uk", "ur", "zh", "unk"]

################################################################################
# parsing/utility functions
################################################################################
''' read in json file containing data set
    returns an array of json entries
'''
def read_json(filename):
    a = []
    for line in open(filename):
        a.append(json.loads(line))
    return a


''' returns a list of languages in the data set
'''
def get_langs(data):
    return [x["lang"] for x in data]


################################################################################
# n-gram Functions
################################################################################
''' generates (word based) n-grams from string input
    where n = size of n-gram
'''
def get_word_ngrams(text, n):
    A = text.split()
    grams = [A[x:x+n] for x in range(len(A))]
    return grams


''' generates (byte based) n-grams from string input
    where n = size of n-gram
'''
def get_byte_ngrams(text, n):
    grams = [text[x:x+n] for x in range(len(text))]
    return grams



################################################################################
# doc-term matrix
################################################################################
''' counts the number of times each n-gram appears in the document
    returns the instance representation in the document-term matrix
'''
def get_feature_vector(v, text):
    return v.transform([text])


''' generates a document-term matrix from data
'''
def get_document_term_matrix(data):
    v = CountVectorizer(analyzer='char')
    corpus = [d["text"] for d in data]
    X = v.fit_transform(corpus)
    Y = v.build_analyzer()
    return [v,X,Y]


################################################################################
# Nearest Prototype Classifier
################################################################################
def nc_eval(tl, dd, dl, v):
    centroids = get_centroids(tl,v)
    labels = []
    for instance in dd:
        label = nearest_prototype(centroids, get_feature_vector(v[0], instance["text"]))
        labels.append(label)

    return accuracy(labels, dl)

def nearest_prototype(centroids, instance):
    dists = dict((key, 0) for key in langs)
    for centroid in centroids:
        dists[centroid] = dist_euclid(instance.toarray()[0], centroids[centroid])

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


''' calculate euclidean distance between two feature vectors
'''
def dist_euclid(v1, v2):
    sum = 0
    for attr1, attr2 in zip(v1, v2):
        sum += (attr1 - attr2) ** 2
    return sum ** 1 / 2



################################################################################
# Naïve Bayes Classifier
################################################################################


################################################################################
# Decision Tree Classifier
################################################################################



################################################################################
# Evaluation Functions
################################################################################
''' accuracy eval
'''
def accuracy(predicted, real):
    T = 0
    N = 0
    for p, r in zip(predicted, real):
        if(p == r): # when we predicted correctly (TP OR TN)
            T += 1
        else: # (FP OR FN)
            N += 1
    return T/(T + N) # equiv to (TP+TN)/(TP+TN+FP+FN)

################################################################################
# main()
################################################################################
if __name__ == "__main__":
    training_data = read_json('in/train.json')
    training_labels = [instance["lang"] for instance in training_data]

    dev_data = read_json('in/dev.json')
    dev_labels = [instance["lang"] for instance in dev_data]

    V = get_document_term_matrix(training_data)

    nc_eval(training_labels, dev_data, dev_labels, V)
