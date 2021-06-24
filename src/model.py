from collections import defaultdict
from gensim import corpora
from gensim import models
from nltk.corpus import stopwords
from tqdm import tqdm

from pathlib import Path
import os, re, string
import pandas as pd

from .IO import loadData, load_model, save_df, save_model, save_tmp, BOW_FILE, DICT_FILE, CORPUS_ID_FILE, get_connection, get_query

def processData(raw_corpus, datasetName):
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
                        texts.append([token.strip() for token in no_nums_string.split(' ') if len(token.strip()) >= min_token_len and token.strip() not in stoplist])
                # Count word frequencies
                frequency = defaultdict(int)
                for text in tqdm(texts, desc='Counting word frequency'):
                        for token in text:
                                frequency[token] += 1
                # Only keep words that appear more than once
                return [[token for token in text if frequency[token] > 1] for text in tqdm(texts, desc='Filtering out unique tokens')]

        processed_corpus = processCorpus(raw_corpus)
        print("Creating dictionary. This may take a few minutes depending on the size of the corpus.")
        dictionary = corpora.Dictionary(processed_corpus)
        dictionary.filter_extremes()
        # Convert original corpus to a bag of words/list of vectors:
        bow_corpus = [dictionary.doc2bow(text) for text in tqdm(processed_corpus, desc='Vectorizing corpus')]
        # Dump processed data to files for faster loading
        save_tmp(datasetName, BOW_FILE, bow_corpus)
        save_tmp(datasetName, DICT_FILE, dictionary)
        return bow_corpus, dictionary

def get_coherence(model, bow_corpus):
    cm = models.coherencemodel.CoherenceModel(model=model, corpus=bow_corpus, coherence='u_mass')
    return cm.get_coherence()

def read_file(settings, dataFile):
    data_type = Path(dataFile).suffix
    encoding = settings['encoding'] if 'encoding' in settings else 'utf-8'
    selected_columns = [settings['corpusFieldName'], settings['dateFieldName']]
    if 'idFieldName' in settings:
        selected_columns.append(settings['idFieldName'])
    if data_type == '.json':
        json_orientation = settings['json_orientation'] if 'json_orientation' in settings else None
        df = pd.read_json(dataFile, orient=json_orientation, encoding=encoding)
        df = df[selected_columns]
    elif data_type == '.csv':
        df = pd.read_csv(dataFile, na_filter=False, usecols=selected_columns)
    else:
        raise Exception
    total_items = df.shape[0]
    df.dropna(inplace=True)
    nan_items = total_items - df.shape[0]
    print(f"Loading {df.shape[0]} items from {dataFile} - {total_items} total items, {nan_items} NaN values were detected and removed.")
    return df

def read_data(settings):
    df = None
    dateFieldName = settings['dateFieldName']
    if 'dataSource' in settings:
        dataSource = os.path.join(os.getcwd(), 'Data', settings['dataSource'])
    if 'db_settings' in settings:
        query = get_query(settings['db_settings'])
        db_connector = get_connection(settings['db_settings'])
        df = pd.read_sql(query, db_connector)
    elif os.path.isdir(dataSource):
        for filename in os.listdir(dataSource):
            file_df = read_file(settings, os.path.join(dataSource, filename))
            if df is None:
                df = file_df
            else:
                df = df.append(file_df)
    else:
        df = read_file(settings, dataSource)
    if 'lang' in settings:
        print(f"Filtering language. Only retaining {settings['lang'][1]} entries.")
        df = df[df[settings['lang'][0]] == settings['lang'][1]]
    print(f"Loaded {df.shape[0]} items.")
    print("Dataset preview:")
    print(df.head())
    print("Fields:")
    print(df.columns.values)
    #Convert to datetime format (useful to filter by date) and round to nearest day
    if 'roundToDay' in settings and settings['roundToDay'] is True:
        df[dateFieldName] = pd.Series(pd.to_datetime(df[dateFieldName])).dt.round("D")
    else:
        df[dateFieldName] = pd.Series(pd.to_datetime(df[dateFieldName]))
    df = df.set_index([dateFieldName])
    df.sort_index(inplace=True)
    return df

def get_data(settings):
    datasetName = settings['datasetName']
    if not settings['reloadData']:
        bow_corpus, dictionary, df = loadData(settings)
    if settings['reloadData'] or bow_corpus is None or dictionary is None or df is None:
        if not settings['reloadData']:
            print("Failure to find processed data. Reloading corpus.")
        print("Processing corpus...")
        df = read_data(settings)
        raw_corpus = df[settings['corpusFieldName']]
        bow_corpus, dictionary = processData(raw_corpus, datasetName)
        # Dump id info to files for faster loading
        save_df(settings, CORPUS_ID_FILE, df.drop(settings['corpusFieldName'], axis=1))
    return bow_corpus, dictionary, df


def create_model(settings, model_type, bow_corpus, dictionary):
    print(f"Training {model_type} model. This may take several minutes depending on the size of the corpus.")
    model = None
    if model_type == 'LDA':
        model = models.LdaModel(bow_corpus, num_topics=settings['numberTopics'], id2word=dictionary, minimum_probability=settings['minimumProbability'])
    elif model_type == 'HDP':
        model = models.HdpModel(bow_corpus, dictionary)
    else:
        print('Invalid model')
        return
    save_model(settings['datasetName'], model, model_type)
    return model

def get_model(settings):
    model = None
    model_type = settings['model_type']
    bow_corpus, dictionary, corpus_df = get_data(settings)
    if not settings['retrainModel']:
        model = load_model(settings['datasetName'], model_type)
    if settings['retrainModel'] or model is None:
        if not settings['retrainModel']:
            print('Failed to load model - recreating.')
        print("Loading model and corpus...")
        model = create_model(settings, model_type, bow_corpus, dictionary)
    print(f"Model coherence score: {get_coherence(model, bow_corpus)}")
    return model, bow_corpus, corpus_df
