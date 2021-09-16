from platform import dist
from gensim import models
import os
from gensim.corpora import dictionary
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..dataManager import load_df, load_tmp, save_df, read_data, save_tmp
from .input import getProcessedData, loadData, loadModel
from .output import getIDsInTopics, saveModel, saveFigures, saveToExcel

class TopicModelNode:
    def __init__(self, project_name, settings) -> None:
        self.DISTRIB_FILE = 'distribDataframe.pkl'
        self.WORDS_FILE = 'wordsDataframe.pkl'
        self.IDS_FILE = 'ids.pkl'
        node = list(settings.keys())[0]
        self.settings = settings[node]
        self.settings['node'] = node
        self.settings['datasetName'] = project_name
        self.MODEL_FILE = f"{self.settings['model']}_model"


    def run(self):
        settings = self.settings
        settings['dataSource'] = settings['file']
        print(f"BEGIN {settings['node']}")
        if 'file' in settings['filters']['in']:
            distributionDF, wordsDF = self.loadFromFile(settings)
        if 'node' in settings['filters']['in']:
            distributionDF, wordsDF = self.loadFromNode(settings)
        if 'folder' in settings['filters']['out']:
            nbFigures = settings['filters']['out']['nbFigures']
            settings['moving_average_size'] = settings['filters']['out']['moving_average_size']
            # saveToExcel(settings, distributionDF, wordsDF)
            saveFigures(settings, distributionDF, wordsDF, n=nbFigures)
        if 'node' in settings['filters']['out']:
            belonging_threshold = 0.3
            topics = settings['filters']['out']['topics']
            filtered_ids = getIDsInTopics(distributionDF, topics, belonging_threshold)
            save_tmp(settings, self.IDS_FILE, filtered_ids)
        print(f"NODE {settings['node']} END")


    def loadFromFile(self, settings):
        print('Attempting to load Dataframes...')
        distributionDF = load_df(settings, self.DISTRIB_FILE)
        wordsDF = load_df(settings, self.WORDS_FILE)
        if settings['reloadData'] or settings['retrainModel'] or (distributionDF is None or wordsDF is None):
            settings['verbose'] = True
            if not settings['reloadData']:
                bow_corpus, dictionary, corpus_df = loadData(settings)
            if settings['reloadData'] or bow_corpus is None or dictionary is None or corpus_df is None:
                if not settings['reloadData']:
                    print("Failure to find processed data. Reloading corpus.")
                print("Fetching data...")
                corpus_df = read_data(settings)
                bow_corpus, dictionary = getProcessedData(settings, corpus_df)
            model = self.get_model(settings, bow_corpus, dictionary)
            print('Calculating Dataframes...')
            distributionDF, wordsDF = self.createMatrix(settings, model, bow_corpus, corpus_df)
        return distributionDF, wordsDF


    def loadFromNode(self, settings):
        print('Attempting to load Dataframes...')
        distributionDF = load_df(settings, self.DISTRIB_FILE)
        wordsDF = load_df(settings, self.WORDS_FILE)
        if settings['reloadData'] or settings['retrainModel'] or (distributionDF is None or wordsDF is None):
            settings['verbose'] = True
            if not settings['reloadData']:
                bow_corpus, dictionary, corpus_df = loadData(settings)
            if settings['reloadData'] or bow_corpus is None or dictionary is None or corpus_df is None:
                if not settings['reloadData']:
                    print("Failure to find processed data. Reloading corpus.")
                print("Fetching data...")
                settings['verbose'] = False
                filtered_ids = []
                corpus_df = read_data(settings)
                filtered_ids = load_tmp(settings, self.IDS_FILE)
                corpus_df = corpus_df[corpus_df[settings['idFieldName']].isin(filtered_ids)]
                print(f"Kept {corpus_df.shape[0]} items.")
                bow_corpus, dictionary = getProcessedData(settings, corpus_df)
            model = self.get_model(settings, bow_corpus, dictionary)
            print('Calculating Dataframes...')
            distributionDF, wordsDF = self.createMatrix(settings, model, bow_corpus, corpus_df)
        return distributionDF, wordsDF


    def getCoherence(self, model, bow_corpus):
        cm = models.coherencemodel.CoherenceModel(model=model, corpus=bow_corpus, coherence='u_mass')
        return cm.get_coherence()


    def createModel(self, settings, model_type, bow_corpus, dictionary):
        print(f"Training {model_type} model. This may take several minutes depending on the size of the corpus.")
        model = None
        if model_type == 'LDA':
            model = models.LdaModel(bow_corpus, num_topics=settings['numberTopics'], id2word=dictionary, minimum_probability=settings['minimumProbability'])
        elif model_type == 'HDP':
            model = models.HdpModel(bow_corpus, dictionary)
        else:
            print('Invalid model')
            return
        saveModel(settings, self.MODEL_FILE, model)
        return model


    def get_model(self, settings, bow_corpus, dictionary):
        model = None
        model_type = settings['model']
        if not settings['retrainModel']:
            model = loadModel(settings, self.MODEL_FILE)
        if settings['retrainModel'] or model is None:
            if not settings['retrainModel']:
                print('Failed to load model - recreating.')
            print("Loading model and corpus...")
            model = self.createModel(settings, model_type, bow_corpus, dictionary)
        print(f"Model coherence score: {self.getCoherence(model, bow_corpus)}")
        return model


    def get_dominant_topics_counts_and_distribution(self, doc_topics):
        """
        Function calculates the counts and relative frequencies for each document represented in doc_topics. See the
        function, get_document_topic_matrix(), to understand the assumed format of doc_topics.

        :param doc_topics: Document-topic matrix in numpy format. Each row represents a document's topic distribution.

        :return: Two numpy arrays:
            1) dominant_topic_counts: array of integers where the ith element is the number of  documents from doc_topics
            that have topic i as their dominant topic.

            2) dominant_topic_dist: array of floats where the ith element is the number of documents from doc_topics with
            dominant topic i divided by the total number of documents in doc_topics.
        """
        # Topic counts start at 0 depending on num topics (or rows) in document-topic matrix.
        dominant_topic_counts = np.zeros(len(doc_topics[0]))
        for topic_dist in tqdm(doc_topics, desc='Fetching dominant topics'):
            # order the topic indices from largest to smallest probability. Without the -1, would get smallest to largest.
            sorted_topic_indices = np.argsort(-1 * topic_dist)
            # The topic index with the largest probability (ie, first in the ordered list)
            dominant_topic = sorted_topic_indices[0]
            # Increment the counter for the corresponding topic in dominant_topic_counts.
            dominant_topic_counts[dominant_topic] += 1
        # Normalize counts to get distribution of dominant topics in the input documents.
        dominant_topic_dist = np.divide(dominant_topic_counts, np.sum(dominant_topic_counts), dtype=np.float64)
        # Ensure that the sum of the distribution is pretty close to 1. Checking precision out to 5 decimal places here.
        assert np.around(np.sum(dominant_topic_dist), 5) == 1.0
        return dominant_topic_counts, dominant_topic_dist


    def print_dominant_topics_by_frequency(self, dominant_topic_counts, dominant_topic_dist):
        """
        Function shows a simple way of printing useful information about the most frequent dominant topics.
        Both of the parameters for this function come from get_dominant_topics_counts_and_distribution() above.
        :param dominant_topic_counts: Array of topic counts such that the ith element is the number of documents
        with dominant topic i.
        :param dominant_topic_dist: Normalized form of dominant_topic_counts such that the ith element is the number of
        documents with dominant topic i divided by the number of documents.
        """
        # Get topic indices ordered from most to least frequent dominant topics.
        ordered_topic_indices = np.argsort(-1 * dominant_topic_counts)
        # Print the topic index, the number of documents with the dominant topic, and the proportion of documents.
        for topic_index in ordered_topic_indices:
            print('topic ' + str(topic_index)
                + ', count=' + str(dominant_topic_counts[topic_index])
                + ', proportion=' + str(np.around(dominant_topic_dist[topic_index], 4)))
        print()

    def get_doc_topic_distributions(self, settings, model, corpus):
        topic_distributions = []
        #    alpha = model.hdp_to_lda()[0]
        #    topics_nos = [x[0] for x in shown_topics ]
        #    weights = [ sum([item[1] for item in shown_topics[topicN][1]]) for topicN in topics_nos ]
        #    df = pd.DataFrame({'topic_id' : topics_nos, 'weight' : weights})
        #FILTER BY DATES HERE
        for doc in tqdm(corpus, desc='Generating topic distribution data'):
            doc_dist = model[doc]
            doc_distribution = np.zeros(settings['numberTopics'], dtype='float64')
            for (topic, val) in doc_dist:
                if topic < settings['numberTopics']:
                        doc_distribution[topic] = val
            topic_distributions.append(doc_distribution)
        topic_distributions = np.asarray(topic_distributions)
        return topic_distributions

    def filterDates(self, settings, bow_corpus, timestamps):
        df = pd.DataFrame(list(zip(timestamps, bow_corpus)), columns =['Date', 'val'])
        df = df.set_index(['Date'])
        df.sort_index(inplace=True)
        #Filter to relevant dates
        if 'start_date' in settings:
            df = df.loc[settings['start_date']:]
        if 'end_date' in settings:
            df = df.loc[:settings['end_date']]
        bow_corpus = df['val'].tolist()
        timestamps = df.index.values
        return bow_corpus, timestamps

    def createMatrix(self, settings, model, bow_corpus, corpusDF):
        timestamps = corpusDF.index.values
        if 'idFieldName' in settings:
            ids = corpusDF[settings['idFieldName']].values
        if 'start_date' in settings or 'end_date' in settings:
            bow_corpus, timestamps = self.filterDates(settings, bow_corpus, timestamps)
        #Topics
        topic_distribution = model.show_topics(num_topics=settings['numberTopics'], num_words=settings['numberWords'],formatted=False)
        topics = []
        for topic_set in topic_distribution:
            topics.append(topic_set[0])

        # Output Topic Distribution and Topic Words
        distribution = self.get_doc_topic_distributions(settings, model, bow_corpus)
        dominant_topic_counts, dominant_topic_dist = self.get_dominant_topics_counts_and_distribution(distribution)
        # print_dominant_topics_by_frequency(dominant_topic_counts, dominant_topic_dist)
        # Set IDs as index if defined then insert timestamp. If not, set timestamp as index
        topic_distrib_df = pd.DataFrame(distribution, index = ids if 'idFieldName' in settings else timestamps, columns=topics)
        if 'idFieldName' in settings:
            topic_distrib_df.insert(0, settings['dateFieldName'], timestamps)
        topic_data = []
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in topic_distribution]
        for topic, words_list in topics_words:
            topic_data.append(words_list + [dominant_topic_counts[topic]] + [np.around(dominant_topic_dist[topic], 4)])
        headers = [f"Word {i}" for i in range(0,len(words_list))] + ["Topic Count", "Distribution"]
        words_df = pd.DataFrame(topic_data, columns=headers)
        # Sort words by descending topic count
        words_df = words_df.sort_values("Topic Count", ascending=False)
        save_df(settings, self.DISTRIB_FILE, topic_distrib_df)
        save_df(settings, self.WORDS_FILE, words_df)
        return topic_distrib_df, words_df
