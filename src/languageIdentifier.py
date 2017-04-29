""" Language Identification System
    Jack Kaloger 2017
    COMP30027 Project 2
"""


import json


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


''' generates a feature vector from a data set
    used for creating a document-term matrix
'''
def get_feature_vector(data):
    A = []
    for x in range(len(data)):
        [A.append(gram) for gram in get_word_ngrams(data[x]["text"], 2)]
    return A


''' counts the number of times each n-gram appears in the document
    returns the instance representation in the document-term matrix
'''
def get_attributes(instance, fv):
    A = [0]*len(fv)
    print(len(fv))
    for attribute in fv:
            A[fv.index(attribute)] += 1
    return A


''' generates a document-term matrix from training data
'''
def get_document_term_matrix(training_data):
    A = []
    fv = get_feature_vector(training_data)
    A.append(fv)
    for instance in training_data:
        A.append(get_attributes(instance, fv))


''' main()
'''
def main():
    training_data = read_json('in/dev.json')
    test_data = read_json('in/dev.json')
    A = get_document_term_matrix(training_data)
    print(A[0])


if __name__ == "__main__":
    main()

