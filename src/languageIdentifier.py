""" Language Identification System
    Jack Kaloger 2017
    COMP30027 Project 2
"""


import json
import collections as c

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
def get_feature_vector(instance):
    return c.Counter(instance)


''' generates a document-term matrix from training data
'''
def get_document_term_matrix(training_data):
    A = []
    for instance in training_data:
        A.append(get_feature_vector(get_byte_ngrams(instance["text"],2)))
    return A


################################################################################
# K-Nearest Neighbour Classifier
################################################################################
''' calculate euclidean distance between two feature vectors
'''
def dist_euclid(v1, v2):
    sum = 0
    for f in v1 + v2:
        sum += (v1[f] - v2[f])**2
    return sum**1/2


''' k-nearest-neighbour evaluation
'''
def knn(training_data, test_data):
    A = get_document_term_matrix(training_data)
    B = get_document_term_matrix(test_data)
    predicted = []
    real = []
    for k,d in enumerate(B):
        dists = []
        for j, i in enumerate(A):
            dists.append([dist_euclid(i, d), training_data[j]["lang"]])
        knn = sorted(dists)
        predicted.append(knn[0][1])
        real.append(test_data[k]["lang"])
        print(knn[0][1] == test_data[k]["lang"])
    print(accuracy(predicted, real))


################################################################################
# Nearest Prototype Classifier
################################################################################
def get_prototypes(training_data, langs):
    #doc-term matrix
    M = get_document_term_matrix(training_data)
    labels = [x["lang"] for x in training_data]
    # our centroid output dict
    centroids = dict((key,c.Counter()) for key in langs)
    count = dict((key, 0) for key in langs)
    # loop through all instances
    for i, instance in enumerate(M):
        centroids[labels[i]] += instance # add to total count
        count[labels[i]] += 1

    for l in centroids:
        for key, val in centroids[l].items():
            centroids[l][key] = val / count[l]

    return centroids


def nearest_prototype(prototypes, test_data):
    T = get_document_term_matrix(test_data)
    for k, d in enumerate(T):
        dists = []
        for key, value in prototypes.items():
            dists.append([dist_euclid(value,d), key])
        print(sorted(dists)[0][1] == test_data[k]["lang"])


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
    training_data = read_json('in/dev.json')
    langs = ["ar", "bg", "de", "en", "es", "fa", "fr", "he", "hi", "it", "ja", "ko", "mr", "ne", "nl", "ru", "th", "uk", "ur", "zh", "unk"]
    test_data = read_json('in/test.json')
    #knn(training_data, test_data)
    P = get_prototypes(training_data, langs)
    print(nearest_prototype(P, test_data))
