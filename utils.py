import json
import nltk


def load_wiki_page(path, name=None):
    wiki_obj = dict()
    for line in open(path, encoding='utf8'):
        obj = json.loads(line)
        if name and name != obj['name']:continue
        wiki_obj[obj['name']] = obj
    return wiki_obj


def clean_genre(name):
    if ' (' in name:
        name = name.split(' (')[0]
    return name


def filter(name):
    punctuations = """'“”’”‐-~!@#$%^&*()_+—*/<>,.[]\/=";{}"""
    if name[0] in punctuations or name[-1] in punctuations:return True
    if '\n' in name:return True
    tokens = nltk.word_tokenize(name)
    if len(tokens) > 7:return True
    filter_tokens = ['in', 'of', 'at', 'and', 'with']
    if tokens[0] in filter_tokens:return True
    return False





