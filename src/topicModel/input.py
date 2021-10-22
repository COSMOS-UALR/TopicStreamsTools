from collections import defaultdict
from gensim import corpora
from gensim import models
from nltk.corpus import stopwords
import re
import string
from tqdm import tqdm

from ..dataManager import fileExists, getFilePath, load_df, load_tmp, save_df, save_tmp

BOW_FILE = 'BOW.obj'
DICT_FILE = 'dictionary.obj'
CORPUS_ID_FILE = 'corpus_id.pkl'
TOKENS_FILE = 'tokens.obj'


def loadModel(settings, file):
    file_path = getFilePath(settings, file)
    if not fileExists(file_path):
        print(f"{file_path} not found.")
        return None
    model = models.LdaModel.load(file_path)
    settings['numberTopics'] = model.num_topics
    return model


def loadData(settings):
    """Load multiple metadata files for topic modeling."""
    bow_corpus = load_tmp(settings, BOW_FILE)
    processed_corpus = load_tmp(settings, TOKENS_FILE)
    dictionary = load_tmp(settings, DICT_FILE)
    corpus_df = load_df(settings, CORPUS_ID_FILE)
    return bow_corpus, dictionary, corpus_df, processed_corpus


def getProcessedData(settings, df):
    """Process data from given corpus df. Returns a bag of word, dictionary, and list of list of tokens."""
    raw_corpus = df[settings['corpusFieldName']]
    bow_corpus, dictionary, processed_corpus = processData(raw_corpus)
    # Dump processed data to files for faster loading
    save_tmp(settings, BOW_FILE, bow_corpus)
    save_tmp(settings, TOKENS_FILE, processed_corpus)
    save_tmp(settings, DICT_FILE, dictionary)
    # Dump id info to files for faster loading
    save_df(settings, CORPUS_ID_FILE, df.drop(settings['corpusFieldName'], axis=1))
    return bow_corpus, dictionary, processed_corpus


def processData(raw_corpus):
    """Process text corpus to obtain bag of words, tokens dictionary, and list of list of tokens."""

    def processCorpus(raw_corpus, min_token_len=3):
        stoplist = set(stopwords.words('english'))
        texts = []
        for document in tqdm(raw_corpus, desc='Normalizing corpus'):
            # lowercase the document string and replace all newlines and tabs with spaces.
            lowercase_string = document.lower().replace('\n', ' ').replace('\t', ' ')
            # replace all punctuation with spaces. (note: this includes punctuation that might occur inside a word).
            punc_pattern = r'[{}]'.format(string.punctuation)
            no_punc_string = re.sub(punc_pattern, ' ', lowercase_string)
            # replace all numbers and non-word chars with spaces. (note: this may not always be a good idea depending on use case).
            no_nums_string = re.sub(r'[\d\W_]', ' ', no_punc_string)
            # split tokens on spaces, trim any space, stop tokens if len() < min_token_len or if in stoplist.
            texts.append([token.strip() for token in no_nums_string.split(' ') if
                          len(token.strip()) >= min_token_len and token.strip() not in stoplist])
        # Count word frequencies
        frequency = defaultdict(int)
        for text in tqdm(texts, desc='Counting word frequency'):
            for token in text:
                frequency[token] += 1
        # Only keep words that appear more than once
        return [[token for token in text if frequency[token] > 1] for text in
                tqdm(texts, desc='Filtering out unique tokens')]
    processed_corpus = processCorpus(raw_corpus)
    print("Creating dictionary. This may take a few minutes depending on the size of the corpus.")
    dictionary = corpora.Dictionary(processed_corpus)
    dictionary.filter_extremes()
    # Convert original corpus to a bag of words/list of vectors:
    bow_corpus = [dictionary.doc2bow(text) for text in tqdm(processed_corpus, desc='Vectorizing corpus')]
    return bow_corpus, dictionary, processed_corpus
