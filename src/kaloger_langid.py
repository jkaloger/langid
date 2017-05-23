""" Short Text Language Identification
    Jack Kaloger 2017
    Project 1 for COMP30027
    
    Thanks for another fun project! I learned a lot in this one.
    My submission classifiers are the SVM, Random Forest and Neural Network
    They can be found after line 292 of this file :)
"""

# some basic libraries for parsing etc
from json import loads
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# visualisation
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# vectorisors for document-term matrices
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# tf-idf
from sklearn.feature_extraction.text import TfidfTransformer

# some basic classifiers
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

# some ensemble learners
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# deep learning stuff
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.utils import compute_class_weight



###############################################################################
# Constants
###############################################################################
langs = ["ar", "bg", "de", "en", "es", "fa", "fr",
         "he", "hi", "it", "ja", "ko", "mr", "ne",
         "nl", "ru", "th", "uk", "ur", "zh", "unk"]

encoder = LabelBinarizer()
encoder.fit(langs)

sources = ["JRC-Acquis", "Debian", "Wikipedia", "twitter"]

stops = [".", "。", "।"]

K = 1000

THRESHOLD = 0.33


################################################################################
# Functions for data set processing
################################################################################
def read_json(filename):
    lines = []
    for line in open(filename):
        line_data = loads(line)
        lines.append(line_data)
    return lines


def get_langs(data):
    return [x["lang"] for x in data]


def get_text(data):
    return [x["text"] for x in data]


def get_ids(data):
    return [x["id"] for x in data]


def lang_clean(data_set):
    for instance in data_set:
        if instance["lang"] not in langs:
            instance["lang"] = "unk"
    return data_set


def text2sentence(data_set):
    new = []
    for instance in data_set:
        sentences = instance["text"].split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence != "":
                new.append({
                    "lang": instance["lang"],
                    "src": instance["src"],
                    "text": sentence
                })
    return new


def src_only(data, src):
    out = []
    for instance in data:
        if instance["src"] == src:
            out.append(instance)

    return out


def src_except(data, src):
    out = []
    for instance in data:
        if instance["src"] != src:
            out.append(instance)

    return out


def tweet_clean(data):
    for instance in data:
        instance["text"] = " ".join(filter(lambda x: x[0] != '#' and x[0] != '@', instance["text"].split()))

    return data


def get_feature_vector(test_data, vect):
    return tf.transform(ptile.transform(vect.transform(test_data[0])))


def get_vectorizer(data_set, an, nr):
    v = HashingVectorizer(non_negative=True, analyzer=an, lowercase=False, ngram_range=nr)
    data_set_text = get_text(data_set)
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


def dectree_eval(vect, dtm, training_data, test_data):
    clf = DecisionTreeClassifier()
    clf.fit(dtm, training_data[1])
    labels = clf.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


################################################################################
# SVM Classifiers
################################################################################
def svm_eval(vect, dtm, training_data, test_data):
    clf = SVC()
    clf.fit(dtm, training_data[1])
    labels = clf.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


def linsvm_eval(vect, dtm, training_data, test_data):
    clf = SVC()
    clf.fit(dtm, training_data[1])
    labels = clf.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


def thresholded_linsvm_eval(vect, dtm, training_data, test_data):
    clf = SVC(kernel='linear', probability=True)
    clf.fit(dtm, training_data[1])
    probs = clf.predict_proba(get_feature_vector(test_data, vect))
    labels = []
    for sample in probs:
        unknown = True
        i = 0
        for prob in sample:
            if prob > THRESHOLD:
                unknown = False
                labels.append(clf.classes_[i])
                break
            i += 1
        if unknown:
            labels.append("unk")
    return accuracy(labels, test_data[1])


################################################################################
# SGD classifier
################################################################################
def sgd_eval(vect, dtm, training_data, test_data):
    sgd = SGDClassifier(loss="hinge", penalty="elasticnet")
    sgd.fit(dtm, training_data[1])
    labels = sgd.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


################################################################################
# Ensemble Classifiers
################################################################################
def voting_eval(vect, dtm, training_data, test_data):
    logit = LogisticRegression(n_jobs=2)
    nb = MultinomialNB()
    ens = VotingClassifier(estimators=[('logit', logit),
                                       ('nb', nb)],
                           voting='soft')
    ens.fit(dtm, training_data[1])
    labels = ens.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


def bagging_eval(vect, dtm, training_data, test_data):
    ens = BaggingClassifier(n_estimators=500)
    ens.fit(dtm, training_data[1])
    labels = ens.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


def rforest_eval(vect, dtm, training_data, test_data):
    ens = RandomForestClassifier(n_estimators=500)
    ens.fit(dtm, training_data[1])
    labels = ens.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


def thresholded_rforest_eval(vect, dtm, training_data, test_data):
    ens = RandomForestClassifier(n_estimators=10)
    ens.fit(dtm, training_data[1])
    probs = ens.predict_proba(get_feature_vector(test_data, vect))
    labels = []
    for sample in probs:
        unknown = True
        i=0
        for prob in sample:
            if prob > THRESHOLD:
                unknown = False
                labels.append(ens.classes_[i])
                break
            i += 1
        if unknown:
            labels.append("unk")
    return stat_eval(labels, test_data[1])


def adaboost_eval(vect, dtm, training_data, test_data):
    ens = AdaBoostClassifier(n_estimators=500)
    ens.fit(dtm, training_data[1])
    labels = ens.predict(get_feature_vector(test_data, vect))

    return stat_eval(labels, test_data[1])


################################################################################
# SUBMISSION CLASSIFIERS
################################################################################
################################################################################
# Linear SVM with thresholding for unknown class
################################################################################
def thresholded_linsvm_predict(vect, dtm, training_data, test_data):
    clf = SVC(kernel='linear', probability=True)
    clf.fit(dtm, training_data[1])
    probs = clf.predict_proba(get_feature_vector(test_data, vect))
    labels = []
    for sample in probs:
        unknown = True
        i = 0
        for prob in sample:
            if prob > THRESHOLD:
                unknown = False
                labels.append(clf.classes_[i])
                break
            i += 1
        if unknown:
            labels.append("unk")

    out = []
    for id, label in zip(test_data[1], labels):
        out.append((id, label))

    return out


################################################################################
# Random Forests with thresholding for unknown class
################################################################################
def thresholded_rforest_predict(vect, dtm, training_data, test_data):
    ens = RandomForestClassifier(n_estimators=50)
    ens.fit(dtm, training_data[1])
    probs = ens.predict_proba(get_feature_vector(test_data, vect))
    labels = []
    for sample in probs:
        unknown = True
        i = 0
        for prob in sample:
            if prob > THRESHOLD:
                unknown = False
                labels.append(ens.classes_[i])
                break
            i += 1
        if unknown:
            labels.append("unk")

    out = []
    for id, label in zip(test_data[1], labels):
        out.append((id, label))

    return out


################################################################################
# My Neural Net Classifier (with thresholding)
################################################################################
def train_gen(X_data, y_data, batch_size):
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


def neural_eval(vect, dtm, training_data, test_data):
    test_data[1] = encoder.fit_transform(test_data[1])
    training_data[1] = encoder.fit_transform(training_data[1])

    model = Sequential()
    model.add(Dense(units=160, kernel_initializer="glorot_uniform", input_shape=dtm.shape[1:], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=80, kernel_initializer="glorot_uniform", activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(units=40, kernel_initializer="glorot_uniform", activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(units=len(langs), kernel_initializer="glorot_uniform", activation='softmax'))

    model.compile(loss='mean_squared_error',
                  optimizer='Adam',
                  metrics=['acc'])

    weights = compute_class_weight(class_weight='balanced', classes=encoder.transform(langs), y=training_data[1])
    weights[encoder.transform(["unk"])] *= 100

    model.fit_generator(train_gen(dtm, training_data[1], 4),
                        steps_per_epoch=len(training_data[1]) / 4,
                        epochs=3,
                        class_weight=weights)

    classes = model.predict_generator(test_gen(get_feature_vector(test_data, vect), 16), len(test_data[1]) / 16)
    '''classes = model.predict_proba(get_feature_vector(test_data, vect).toarray())
    labels = []
    for sample in classes:
        p = sample.argmax(axis=-1)
        if p > THRESHOLD:
                labels.append(p)
        else:
            labels.append(encoder.transform("unk"))'''
    return stat_eval(encoder.inverse_transform(classes), encoder.inverse_transform(test_data[1]))


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
    if ((B*B*precision) + recall) <= 0:
        return 0
    return (1 + B*B) * ((precision * recall) / ((B*B*precision) + recall))


################################################################################
# MAIN
################################################################################
if __name__ == "__main__":
    # load development data
    dev_data = read_json('in/dev.json')

    training_data = read_json('in/train.json')
    training_data = src_except(training_data, "JRC-Acquis")
    #training_data = src_only(training_data, "twitter")
    training_data = lang_clean(training_data)
    training_data.append({"lang": "unk",
                          "src": "internal",
                          "text": "  "})

    print("getting doc-term matrix...")
    # get document term matrix
    V = get_vectorizer(training_data, "char_wb", (1, 2))

    print("selecting features...")
    # apply feature selection
    ptile = SelectKBest(score_func=chi2, k=K)
    dtm_new = ptile.fit_transform(V[1], get_langs(training_data))

    print("applying tf-idf...")
    # apply tf-idf
    tf = TfidfTransformer()
    dtm_new = tf.fit_transform(dtm_new)


    '''print(neural_eval(V[0], dtm_new, [get_text(training_data), get_langs(training_data)],
                       [get_text(dev_data), get_langs(dev_data)]))'''

    for i in range(5,95,5):
        THRESHOLD = i/100
        print("evaluating... %d" % i)

        print(thresholded_linsvm_eval(V[0], dtm_new, [get_text(training_data), get_langs(training_data)],
                                      [get_text(dev_data), get_langs(dev_data)]))

    '''predictions = thresholded_rforest_predict(V[0], dtm_new, [get_text(training_data), get_langs(training_data)],
                       [get_text(test_data), get_ids(test_data)])

    f = open('../out/dtree', 'w')
    f.write("id,prediction\n")
    for pred in predictions:
        f.write("%s,%s\n" % (pred[0], pred[1]))
    f.close()'''
