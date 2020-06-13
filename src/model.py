from collections import defaultdict
from gensim import corpora
from gensim import models
from nltk.corpus import stopwords
from pandas.io.json import json_normalize  
from tqdm import tqdm

from pathlib import Path
import json, os, re, string
import pandas as pd

from .IO import loadData, loadFiles, load_tmp, save_model, save_tmp, BOW_FILE, DICT_FILE, ID_FILE

def processData(ids, raw_corpus, datasetName):
        def processCorpus(raw_corpus, min_token_len=3):
                stoplist = set(stopwords.words('english'))
                texts = []
                print("Normalizing corpus...")
                for document in tqdm(raw_corpus):
                        # lowercase the document string and replace all newlines and tabs with spaces.
                        lowercase_string = document.lower().replace('\n', ' ').replace('\t', ' ')
                        # replace all punctuation with spaces. (note: this includes punctuation that might occur inside a word).
                        punc_pattern = r'[{}]'.format(string.punctuation)
                        no_punc_string = re.sub(punc_pattern, ' ', lowercase_string)
                        # replace all numbers and non-word chars with spaces. (note: this may not always be a good idea depending on use case).
                        no_nums_string = re.sub(r'[\d\W_]', ' ', no_punc_string)
                        # split tokens on spaces, trim any space, stop tokens if len() < min_token_len or if in stoplist.
                        texts.append([token.strip() for token in no_nums_string.split(' ') if len(token.strip()) >= min_token_len and token.strip() not in stoplist])
                # Count word frequencies
                print("Counting word frequency...")
                frequency = defaultdict(int)
                for text in tqdm(texts):
                        for token in text:
                                frequency[token] += 1
                # Only keep words that appear more than once
                print("Filtering out unique tokens...")
                return [[token for token in text if frequency[token] > 1] for text in tqdm(texts)]

        processed_corpus = processCorpus(raw_corpus)
        print("Creating dictionary. This may take a few minutes depending on the size of the corpus.")
        dictionary = corpora.Dictionary(processed_corpus)
        dictionary.filter_extremes()
        # Convert original corpus to a bag of words/list of vectors:
        print("Vectorizing corpus...")
        bow_corpus = [dictionary.doc2bow(text) for text in tqdm(processed_corpus)]
        # Dump corpus, dictionary, and IDs to physical files for faster loading
        save_tmp(datasetName, BOW_FILE, bow_corpus)
        save_tmp(datasetName, DICT_FILE, dictionary)
        save_tmp(datasetName, ID_FILE, ids)
        return bow_corpus, dictionary, ids

def read_data(dataFile, corpusFieldName, idFieldName):
    data_type = Path(dataFile).suffix
    df = None
    ids = None
    if data_type == '.json':
        with open(dataFile, encoding="UTF-8") as f:
            dataObject = json.load(f, strict=False)
            df = pd.DataFrame(dataObject)
            df = df[df['language'] == "English"]
    elif data_type == '.csv':
        df = pd.read_csv(dataFile, na_filter=False)
        ids = df[idFieldName]
    else:
        raise Exception
    print(f"Loaded {df.shape[0]} items.")
    print("Dataset preview:")
    print(df.head())
    print("Fields:")
    print(df.columns.values)
    #Convert to datetime format (useful to filter by date) and round to nearest day
    df[idFieldName] = pd.Series(pd.to_datetime(df[idFieldName])).dt.round("D")
    df = df.set_index([idFieldName])
    df.sort_index(inplace=True)
    ids = df.index.values
    raw_corpus = df[corpusFieldName]
    return raw_corpus, ids
    
def create_model(settings):

    bow_corpus, dictionary, ids = loadData(settings['datasetName'])

    if settings['reloadData'] or bow_corpus is None or dictionary is None or ids is None:
        if not settings['reloadData']:
            print("Failure to find processed data. Reloading corpus.")
        print("Processing model and corpus...")
        dataFile = os.getcwd() + '\\Data\\' + settings['filePath']
        raw_corpus, ids = read_data(dataFile, settings['corpusFieldName'], settings['idFieldName'])
        bow_corpus, dictionary, ids = processData(ids, raw_corpus, settings['datasetName'])

    print("Training model. This may take several minutes depending on the size of the corpus.")
    model = models.LdaModel(bow_corpus, num_topics=settings['numberTopics'], id2word=dictionary, minimum_probability=settings['minimumProbability'])
    save_model(settings['datasetName'], model)

    return model, bow_corpus, ids

def get_model(settings):
    if settings['reloadData'] or settings['retrainModel']:
        return create_model(settings)
    else:
        print("Loading model and corpus...")
        model, bow_corpus, ids = loadFiles(settings['datasetName'])
        if model is None or bow_corpus is None or ids is None:
            print("Failed to load files - recreating model")
            return create_model(settings)    
