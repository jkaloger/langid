# Language Identification System
# Jack Kaloger 2017
# COMP30027 Project 2

import json


def read_json(filename):
    a = []
    for line in open(filename):
        a.append(json.loads(line))
    return a


def main():
    data = read_json('in/dev.json')
    print(data)


if __name__ == "__main__":
    main()
