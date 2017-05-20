""" Short Text Language Identification
    Jack Kaloger 2017
    Project 1 for COMP30027
"""

# some basic libraries for parsing etc
from json import loads
from random import randint
import math
import numpy as np
import itertools
from sklearn.preprocessing import LabelBinarizer

# vectorisors for document-term matrices
from sklearn.feature_extraction.text import HashingVectorizer

# some basic classifiers
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# some ensemble learners
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier

# tensorflow for deep learning
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.utils.np_utils import to_categorical



###############################################################################
# Constants
###############################################################################
langs = ["ar", "bg", "de", "en", "es", "fa", "fr",
         "he", "hi", "it", "ja", "ko", "mr", "ne",
         "nl", "ru", "th", "uk", "ur", "zh", "unk"]


################################################################################
# Functions for data set processing
################################################################################
def read_json(filename):
    data_set = []
    labels = []
    for line in open(filename):
        line_data = loads(line)
        if(line_data["lang"] == "unk"):
            continue
        data_set.append(line_data["text"])
        labels.append(line_data["lang"])
        if randint(1, 4) == 4:
            continue
    return [data_set, labels]


def get_langs(data):
    return [x["lang"] for x in data]


def lang_clean(data):
    i=0
    for instance in data:
        if instance not in langs:
            data[i] = "unk"
        i += 1
    return data


def tweet_clean(data):
    for instance in data:
        instance["text"] = " ".join(filter(lambda x: x[0] != '#' and x[0] != '@', instance["text"].split()))

    return data


def get_feature_vector(test_data, vect):
    return vect.transform(test_data[0])


def get_vectorizer(data_set, an, nr):
    v = HashingVectorizer(non_negative=True, analyzer=an, lowercase=False, ngram_range=nr)
    data_set_text = data_set[0]
    dtm = v.fit_transform(data_set_text)
    return [v, dtm]


################################################################################
# Baseline Classifiers (initial testing)
################################################################################
def nc_eval(vect, dtm, training_data, test_data):
    clf = NearestCentroid()
    clf.fit(dtm, training_data[1])
    labels = clf.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


def nb_eval(vect, dtm, training_data, test_data):
    clf = MultinomialNB()
    clf.fit(dtm, training_data[1])
    labels = clf.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


def svm_eval(vect, dtm, training_data, test_data):
    clf = SVC()
    clf.fit(dtm, training_data[1])
    labels = clf.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])

def linsvm_eval(vect, dtm, training_data, test_data):
    clf = LinearSVC()
    clf.fit(dtm, training_data[1])
    labels = clf.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


def dectree_eval(vect, dtm, training_data, test_data):
    clf = DecisionTreeClassifier()
    clf.fit(dtm, training_data[1])
    labels = clf.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


################################################################################
# Ensemble Classifiers
################################################################################
def voting_eval(vect, dtm, training_data, test_data):
    svm = LinearSVC()
    nc = NearestCentroid()
    dt = DecisionTreeClassifier()
    nb = MultinomialNB()
    ens = VotingClassifier(estimators=[('svm', svm),
                                       ('nc', nc),
                                       ('dt', dt),
                                       ('nb', nb)],
                           voting='hard')
    ens.fit(dtm, training_data[1])
    labels = ens.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


def adaboost_eval(vect, dtm, training_data, test_data):
    ens = AdaBoostClassifier(n_estimators=50)
    ens.fit(dtm, training_data[1])
    labels = ens.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


################################################################################
# My Neural Net Classifier (where the fun begins)
################################################################################
def nn_batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter = 0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        y_batch = y_data[index_batch]
        counter += 1
        yield np.array(X_batch),y_batch
        if (counter > number_of_batches):
            counter = 0

def test_gen(X_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter = 0
    index = np.arange(np.shape(X_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        counter += 1
        yield np.array(X_batch)
        if (counter > number_of_batches):
            counter = 0

################################################################################
# Evaluation functions
################################################################################
def stat_eval(predicted, real):
    acc = accuracy(predicted, real)
    prec = macro_precision(predicted, real)
    recl = macro_recall(predicted, real)
    fscr = f_score(1, prec, recl)

    return [acc, prec, recl, fscr]


def oneR(real):
    sum = 0
    for lang in real:
        if lang == "en":
            sum += 1

    return sum/len(real)


def accuracy(predicted, real):
    t = 0
    n = 0
    for p, r in zip(predicted, real):
        if p == r:  # when we predicted correctly (TP OR TN)
            t += 1
        else:  # (FP OR FN)
            n += 1
    return t/(t + n)  # equiv to (TP+TN)/(TP+TN+FP+FN)


def create_confusion_matrix(predicted, real):
    M = dict((outer, dict((inner, 0) for inner in langs)) for outer in langs) # Predicted across, Real down
    for p, r in zip(predicted, real):
        if p in langs and r in langs:
            M[p][r] += 1
    return M


def precision(M, c):
    if M[c][c] == 0:
        return 0
    FP = 0
    for lang in langs:
        if lang != c:
            FP += M[c][lang]
    if FP == 0:
        return 1
    return M[c][c]/(M[c][c]+FP) # TP / TP + FP


def macro_precision(predicted, real):
    M = create_confusion_matrix(predicted, real)
    sum = 0
    for lang in langs:
        sum += precision(M, lang)

    return sum/len(langs)


def recall(M, c):
    if M[c][c] == 0:
        return 0
    FN = 0
    for lang in langs:
        if lang != c:
            FN += M[lang][c]
    if FN == 0:
        return 1
    return M[c][c]/(M[c][c]+FN) # TP / TP + FP


def macro_recall(predicted, real):
    M = create_confusion_matrix(predicted, real)
    sum = 0
    for lang in langs:
        sum += recall(M, lang)

    return sum / len(langs)


def f_score(B, precision, recall):
    return (1 + B*B) * ((precision * recall) / ((B*B*precision) + recall))


################################################################################
# MAIN
################################################################################
if __name__ == "__main__":
    # training data
    training_data = read_json('in/train.json')
    print("training data parsed")
    print(len(training_data[0]))
    training_data[1] = lang_clean(training_data[1])
    # dev data
    dev_data = read_json('in/dev.json')
    print("dev data parsed")

    V = get_vectorizer(training_data, "word", (1,1))



    # document term matrix
    '''analyzers = ["word", "char", "char_wb"]
    ngrams = [(1,1), (1,2), (2,2)]

    for a in analyzers:
        for n in ngrams:
            print(a)
            print(n)
            V = get_vectorizer(training_data, a, n)

            vect = V[0]
            dtm = V[1]
            print("dtm parsed")

            print("Naïve Bayes, Nearest Centroid, LSVM, Decision Trees, Voting Ensemble, Adaboost Ensemble")
            print(nb_eval(vect, dtm, training_data, dev_data))
            print(nc_eval(vect, dtm, training_data, dev_data))
            print(linsvm_eval(vect, dtm, training_data, dev_data))
            print(dectree_eval(vect, dtm, training_data, dev_data))
            print(voting_eval(vect, dtm, training_data, dev_data))
            print(adaboost_eval(vect, dtm, training_data, dev_data))'''

    encoder = LabelBinarizer()
    training_data[1] = encoder.fit_transform(training_data[1])
    dev_data[1] = encoder.fit_transform(dev_data[1])

    model = Sequential()
    print(V[1].shape)
    model.add(Dense(units=64, input_shape=V[1].shape[1:]))
    model.add(Activation('relu'))
    model.add(Dense(units=20))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

    # model.fit(V[1].toarray(), training_data[1], batch_size=32, epochs=10)
    model.fit_generator(nn_batch_generator(V[1], training_data[1], 32), steps_per_epoch=32, epochs=3)

    classes = model.predict_generator(test_gen(get_feature_vector(dev_data, V[0]), 32), 188)

    print(classes)
