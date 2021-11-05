from gensim import models
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..dataManager import load_df, load_tmp, save_df, read_data, save_tmp
from .input import getProcessedData, loadData, loadModel
from .output import getIDsInTopics, saveCoherencePlot, saveInteractivePage, saveModel, saveFigures, saveToExcel

DEFAULT_COHERENCE = 'c_v'

class TopicModelNode:
    def __init__(self, project_name, settings) -> None:
        self.DISTRIB_FILE = 'distribDataframe.pkl'
        self.WORDS_FILE = 'wordsDataframe.pkl'
        self.IDS_FILE = 'ids.pkl'
        node = list(settings.keys())[0]
        self.settings = settings[node]
        self.settings['node'] = node
        self.settings['datasetName'] = project_name
        if 'dataSource' in self.settings:
            self.settings['dataSource'] = self.settings['dataSource']
        elif 'file' in self.settings:
            self.settings['dataSource'] = self.settings['file']
        self.settings['coherence_measure'] = self.settings['coherence_measure'] if 'coherence_measure' in self.settings else DEFAULT_COHERENCE
        self.MODEL_FILE = f"{self.settings['model']}_model"


    def run(self):
        settings = self.settings
        print(f"BEGIN {settings['node']}")
        load_from_node = 'node' in settings['filters']['in']
        distributionDF, wordsDF, model, bow_corpus, dictionary = self.load(settings, from_node=load_from_node)
        if 'folder' in settings['filters']['out']:
            nbFigures = settings['filters']['out']['nbFigures']
            settings['moving_average_size'] = settings['filters']['out']['moving_average_size']
            saveToExcel(settings, distributionDF, wordsDF)
            saveFigures(settings, distributionDF, wordsDF, n=nbFigures)
            saveInteractivePage(settings, model, bow_corpus, dictionary)
        if 'node' in settings['filters']['out']:
            belonging_threshold = 0.3
            topics = settings['filters']['out']['topics']
            filtered_ids = getIDsInTopics(distributionDF, topics, belonging_threshold)
            save_tmp(settings, self.IDS_FILE, filtered_ids)
        print(f"NODE {settings['node']} END")


    def load(self, settings, from_node=False):
        """Will attempt to load processed file, or regenerate if retrainData or retrainModel are True."""
        print('Attempting to load Dataframes...')
        settings['verbose'] = True
        if not settings['reloadData']:
            bow_corpus, dictionary, corpus_df, processed_corpus = loadData(settings)
        if settings['reloadData'] or bow_corpus is None or dictionary is None or corpus_df is None or processed_corpus is None:
            if not settings['reloadData']:
                print("Failure to find processed data. Reloading corpus.")
            print("Fetching data...")
            if from_node:
                settings['verbose'] = False
                filtered_ids = []
                corpus_df = read_data(settings)
                filtered_ids = load_tmp(settings, self.IDS_FILE)
                corpus_df = corpus_df[corpus_df[settings['idFieldName']].isin(filtered_ids)]
                print(f"Kept {corpus_df.shape[0]} items.")
            else:
                corpus_df = read_data(settings)
            bow_corpus, dictionary, processed_corpus = getProcessedData(settings, corpus_df)
        model = loadModel(settings, self.MODEL_FILE)
        distributionDF = load_df(settings, self.DISTRIB_FILE)
        wordsDF = load_df(settings, self.WORDS_FILE)
        if settings['reloadData'] or settings['retrainModel'] or model is None or distributionDF is None or wordsDF is None:
            model = self.getModel(settings, bow_corpus, dictionary, processed_corpus)
            print('Calculating Dataframes...')
            distributionDF, wordsDF = self.createMatrix(settings, model, bow_corpus, corpus_df)
        return distributionDF, wordsDF, model, bow_corpus, dictionary


    def getCoherenceModel(self, model, coherence_measure, bow_corpus, processed_corpus, dictionary):
        if coherence_measure == 'c_v':
            return models.coherencemodel.CoherenceModel(model=model, texts=processed_corpus, dictionary=dictionary, coherence='c_v')
        return models.coherencemodel.CoherenceModel(model=model, corpus=bow_corpus, coherence='u_mass')


    def createModel(self, model_type, bow_corpus, dictionary, processed_corpus, number_topics):
        print(f"Training {model_type} model. This may take several minutes depending on the size of the corpus.")
        model = None
        iterations = self.settings['iterations'] if 'iterations' in self.settings else 50
        passes = self.settings['passes'] if 'passes' in self.settings else 1
        if model_type == 'LDA':
            model = models.LdaModel(bow_corpus, num_topics=number_topics, id2word=dictionary, passes=passes, alpha='auto', iterations=iterations, minimum_probability=self.settings['minimumProbability'])
        elif model_type == 'LDA-Mallet':
            os.environ['MALLET_HOME'] = self.settings['mallet_path']
            mallet_path = os.path.join(self.settings['mallet_path'], 'bin', 'mallet')
            model = models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus, num_topics=number_topics, id2word=dictionary, iterations=iterations)
        elif model_type == 'HDP':
            model = models.HdpModel(bow_corpus, dictionary)
        else:
            print('Invalid model')
            return
        coherence_model = self.getCoherenceModel(model, self.settings['coherence_measure'], bow_corpus, processed_corpus, dictionary)
        print(f"Model coherence score: {coherence_model.get_coherence()}")
        return model


    def getModel(self, settings, bow_corpus, dictionary, processed_corpus):
        if 'optimize_model' in settings and settings['optimize_model']:
            model = self.optimizeModel(dictionary, bow_corpus, processed_corpus)
        else:
            model = self.createModel(settings['model'], bow_corpus, dictionary, processed_corpus, self.settings['numberTopics'])
        saveModel(settings, self.MODEL_FILE, model)
        return model


    def optimizeModel(self, dictionary, bow_corpus, processed_corpus):
        """Compute coherence for various number of topics."""
        coherence_values = []
        model_list = []
        topics_range = range(2, 40, 4)
        for num_topics in topics_range:
            model = self.createModel(self.settings['model'], bow_corpus, dictionary, processed_corpus, num_topics)
            model_list.append(model)
            coherence_model = self.getCoherenceModel(model, self.settings['coherence_measure'], bow_corpus, processed_corpus, dictionary)
            coherence_values.append(coherence_model.get_coherence())
        saveCoherencePlot(self.settings, coherence_values, topics_range, self.settings['coherence_measure'])
        best_result_index = coherence_values.index(max(coherence_values))
        optimal_model = model_list[best_result_index]
        optimal_number_topics = topics_range[best_result_index]
        print(f'{optimal_number_topics} topics gives the highest coherence score of {coherence_values[best_result_index]}')
        self.settings['numberTopics'] = optimal_number_topics
        return optimal_model


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
        # Set IDs as index if defined then insert timestamp. If not, set timestamp as index
        topic_distrib_df = pd.DataFrame(distribution, index = ids if 'idFieldName' in settings else timestamps, columns=topics)
        if 'idFieldName' in settings and 'dateFieldName' in settings:
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
