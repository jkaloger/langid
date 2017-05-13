""" Short Text Language Identification
    Jack Kaloger 2017
    Project 1 for COMP30027
"""


from json import loads
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.spatial.distance import cosine


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
    for line in open(filename):
        data_set.append(loads(line))
    return data_set


def get_langs(data):
    return [x["lang"] for x in data]


def tweet_clean(data):
    for instance in data:
        instance["text"] = " ".join(filter(lambda x: x[0] != '#' and x[0] != '@', instance["text"].split()))

    return data


def get_feature_vector(instance, v):
    return v.transform(instance["text"])


def get_document_term_matrix(data_set):
    v = HashingVectorizer(analyzer="char", lowercase=False)
    return v.transform(data_set)