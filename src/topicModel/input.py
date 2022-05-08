from collections import defaultdict
import errno
from gensim import corpora
from gensim import models
from nltk.corpus import stopwords
import os
import re
import string
from tqdm import tqdm

from ..dataManager import fileExists, getFilePath, load_tmp, save_tmp, get_connection, fetchData, quoteList, countQueryItems

BOW_FILE = 'BOW.obj'
DICT_FILE = 'dictionary.obj'
TOKENS_FILE = 'tokens.obj'
STOPWORDS_FILE = 'stopwords.txt'


def loadModel(settings, file):
    file_path = getFilePath(settings, file)
    if not fileExists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    if settings['model'] == 'LDA':
        model = models.LdaModel.load(file_path)
    if settings['model'] == 'HDP':
        model = models.HdpModel.load(file_path)
    if settings['model'] == 'LDA-Mallet':
        model = models.wrappers.LdaMallet.load(file_path)
    settings['numberTopics'] = model.num_topics
    return model


def loadProcessedData(settings):
    """Load multiple metadata files for topic modeling."""
    bow_corpus = load_tmp(settings, BOW_FILE)
    processed_corpus = load_tmp(settings, TOKENS_FILE)
    dictionary = load_tmp(settings, DICT_FILE)
    return bow_corpus, dictionary, processed_corpus


def getProcessedData(settings, df):
    """Process data from given corpus df. Returns a bag of word, dictionary, and list of list of tokens."""
    raw_corpus = df[settings['corpusFieldName']]
    bow_corpus, dictionary, processed_corpus = processData(raw_corpus)
    # Dump processed data to files for faster loading
    save_tmp(settings, BOW_FILE, bow_corpus)
    save_tmp(settings, TOKENS_FILE, processed_corpus)
    save_tmp(settings, DICT_FILE, dictionary)
    return bow_corpus, dictionary, processed_corpus


def processData(raw_corpus):
    """Process text corpus to obtain bag of words, tokens dictionary, and list of list of tokens."""

    def processCorpus(raw_corpus, min_token_len=3):
        stoplist = set(stopwords.words('english'))
        if fileExists(STOPWORDS_FILE):
            stoplist.update(set(line.strip() for line in open(STOPWORDS_FILE)))
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


def queryChannelData(settings, channel_ids):
    """Return video data belonging to channel(s)."""
    db_connector = get_connection(settings['filters']['in']['db_settings'])
    CHUNKSIZE = 10000
    video_table = 'videos'
    columns = ['video_id', 'published_date', 'video_title']
    query_filter = f'FROM {video_table} WHERE channel_id IN ({",".join(quoteList(channel_ids))})'
    # query_filter += f' AND published_date >= "2019-01-01" ORDER BY published_date ASC'
    query = f'SELECT {",".join(columns)} ' + query_filter
    query_count = f'SELECT COUNT(video_id) ' + query_filter
    total_videos = countQueryItems(db_connector, query_count)
    video_df = fetchData(db_connector, query, video_table, columns, CHUNKSIZE, total=total_videos)
    return video_df
