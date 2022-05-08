from gensim import models
import numpy as np
import os
import pandas as pd
import pyLDAvis
import queue, threading
from tqdm import tqdm

from ..dataManager import load_df, save_df, save_tmp
from .baseModel import BaseModel
from .input import getProcessedData, loadProcessedData, loadModel
from .output import getIDsInTopics, saveCoherencePlot, saveInteractivePage, saveModel, saveFigures, saveToExcel

DEFAULT_COHERENCE = 'c_v'

class GensimModel(BaseModel):

    def __init__(self, settings):
        super().__init__(settings)
        self.bow_corpus = None
        self.dictionary = None
        self.settings['coherence_measure'] = self.settings['coherence_measure'] if 'coherence_measure' in self.settings else DEFAULT_COHERENCE
    

    def processData(self):
        """Will attempt to load processed file, or regenerate if they are missing or reloadData is True."""
        settings = self.settings
        settings['verbose'] = True
        if not settings['reloadData']:
            try:
                self.bow_corpus, self.dictionary, self.processed_corpus = loadProcessedData(settings)
            except FileNotFoundError as e:
                print(str(e))
                self.settings['reloadData'] = True
                self.loadData()
        if self.settings['reloadData']:
            self.bow_corpus, self.dictionary, self.processed_corpus = getProcessedData(settings, self.corpus_df)
        

    def train(self):
        """Will attempt to load the model and its associated data, will train and create if they cannot be found or retrainModel is True."""
        print('Attempting to load Dataframes...')
        settings = self.settings
        try:
            self.model = loadModel(settings, self.MODEL_FILE)
            self.distributionDF = load_df(settings, self.DISTRIB_FILE)
            self.wordsDF = load_df(settings, self.WORDS_FILE)
        except FileNotFoundError as e:
            print(str(e))
            settings['retrainModel'] = True
        if settings['reloadData'] or settings['retrainModel']:
            self.model = self.getModel(settings, self.bow_corpus, self.dictionary, self.processed_corpus)
            print('Calculating Dataframes...')
            topic_distribution = self.model.show_topics(num_topics=settings['numberTopics'], num_words=settings['numberWords'],formatted=False)
            theta = self.getTheta(settings, self.model, self.bow_corpus)
            self.distributionDF = self.getThetaDF(settings, self.bow_corpus, self.corpus_df, topic_distribution, theta)
            save_df(settings, self.DISTRIB_FILE, self.distributionDF)
            self.wordsDF = self.getTopTopicWords(settings, topic_distribution, theta)


    def output(self):
        """Will write to disk the results of the model's findings."""
        settings = self.settings
        out = {}
        if 'folder' in settings['filters']['out']:
            nbFigures = settings['filters']['out']['nbFigures']
            settings['moving_average_size'] = settings['filters']['out']['moving_average_size']
            saveToExcel(settings, self.distributionDF, self.wordsDF)
            saveFigures(settings, self.distributionDF, self.wordsDF, n=nbFigures)
            saveInteractivePage(settings, self.getLDAVisPreparedData())
        if 'node' in settings['filters']['out']:
            topics = settings['filters']['out']['topics']
            filtered_ids = getIDsInTopics(self.distributionDF, topics, self.belonging_threshold)
            # save_tmp(settings, self.IDS_FILE, filtered_ids)
            out['video_ids'] = filtered_ids
        return out


    def getModel(self, settings, bow_corpus, dictionary, processed_corpus):
        """Will create and optimize a model, or create a model from given parameters."""
        if 'optimize_model' in settings and settings['optimize_model']:
            model = self.optimizeModel(dictionary, bow_corpus, processed_corpus)
        else:
            model = self.createModel(settings['model'], bow_corpus, dictionary, processed_corpus, self.settings['numberTopics'])
        saveModel(settings, self.MODEL_FILE, model)
        return model


    def getCoherenceModel(self, model, coherence_measure, bow_corpus, processed_corpus, dictionary):
        if coherence_measure == 'c_v':
            return models.coherencemodel.CoherenceModel(model=model, texts=processed_corpus, dictionary=dictionary, coherence='c_v')
        return models.coherencemodel.CoherenceModel(model=model, corpus=bow_corpus, coherence='u_mass')


    def multithreadedCreateModel(self, q, bow_corpus, dictionary, processed_corpus, model_list):
        args = q.get()
        index = args[0]
        num_topics = args[1]
        model = self.createModel(self.settings['model'], bow_corpus, dictionary, processed_corpus, num_topics)
        model_list[index] = model
        q.task_done()


    def optimizeModel(self, dictionary, bow_corpus, processed_corpus):
        """Compute coherence for various number of topics."""
        topics_range = range(2, 40, 4)
        coherence_values = []
        model_list = [None for _ in topics_range]
        if 'multithreading' in self.settings and self.settings['multithreading']:
            q = queue.Queue()
            # for _ in range(multiprocessing.cpu_count()):
            for i, num_topics in enumerate(topics_range):
                thread = threading.Thread(target=self.multithreadedCreateModel, args=(q, bow_corpus, dictionary, processed_corpus, model_list), daemon=True)
                thread.start()
            for i, num_topics in enumerate(topics_range):
                q.put((i, num_topics))
            q.join()
        else:
            for i, num_topics in enumerate(topics_range):
                model_list[i] = self.createModel(self.settings['model'], bow_corpus, dictionary, processed_corpus, num_topics)
        for model in model_list:
            coherence_values.append(self.getCoherenceModel(model, self.settings['coherence_measure'], bow_corpus, processed_corpus, dictionary).get_coherence())
        saveCoherencePlot(self.settings, coherence_values, topics_range, self.settings['coherence_measure'])
        best_result_index = coherence_values.index(max(coherence_values))
        optimal_model = model_list[best_result_index]
        optimal_number_topics = topics_range[best_result_index]
        print(f'{optimal_number_topics} topics gives the highest coherence score of {coherence_values[best_result_index]}')
        self.settings['numberTopics'] = optimal_number_topics
        return optimal_model


    def getTheta(self, settings, model, corpus):
        """Get topic distributions for each document."""
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
        timestamps = df[settings['dateFieldName']].values
        return bow_corpus, timestamps


    def getThetaDF(self, settings, bow_corpus, corpusDF, topic_distribution, theta):
        """Get topic distributions dataframe for each document, with document id or timestamped as index."""
        timestamps = corpusDF[settings['dateFieldName']].values
        if 'idFieldName' in settings:
            ids = corpusDF[settings['idFieldName']].values
        if 'start_date' in settings or 'end_date' in settings:
            bow_corpus, timestamps = self.filterDates(settings, bow_corpus, timestamps)
        topics = [topic_set[0] for topic_set in topic_distribution]
        # Set IDs as index if defined then insert timestamp. If not, set timestamp as index
        topic_distrib_df = pd.DataFrame(theta, index = ids if 'idFieldName' in settings else timestamps, columns=topics)
        if 'idFieldName' in settings and 'dateFieldName' in settings:
            topic_distrib_df.insert(0, settings['dateFieldName'], timestamps)
        return topic_distrib_df
    
    def getTopTopicWords(self, settings, topic_distribution, theta):
        """Get top words for each topic."""
        topic_data = []
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in topic_distribution]
        dominant_topic_counts, dominant_topic_dist = self.get_dominant_topics_counts_and_distribution(theta)
        for topic, words_list in topics_words:
            topic_data.append(words_list + [dominant_topic_counts[topic]] + [np.around(dominant_topic_dist[topic], 4)])
        headers = [f"Word {i}" for i in range(0,len(words_list))] + ["Topic Count", "Distribution"]
        words_df = pd.DataFrame(topic_data, columns=headers)
        # Sort words by descending topic count
        words_df = words_df.sort_values("Topic Count", ascending=False)
        save_df(settings, self.WORDS_FILE, words_df)
        return words_df


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

    def getLDAVisPreparedData(self):
        return pyLDAvis.gensim.prepare(self.model, self.bow_corpus, self.dictionary)


class LDAModel(GensimModel):

    def createModel(self, model_type, bow_corpus, dictionary, processed_corpus, number_topics):
        print(f"Training {model_type} model. This may take several minutes depending on the size of the corpus.")
        model = None
        iterations = self.settings['iterations'] if 'iterations' in self.settings else 50
        passes = self.settings['passes'] if 'passes' in self.settings else 1
        model = models.LdaModel(bow_corpus, num_topics=number_topics, id2word=dictionary, passes=passes, alpha='auto', iterations=iterations, minimum_probability=self.settings['minimumProbability'])
        coherence_model = self.getCoherenceModel(model, self.settings['coherence_measure'], bow_corpus, processed_corpus, dictionary)
        print(f"Model coherence score: {coherence_model.get_coherence()}")
        return model

class LDAMalletModel(GensimModel):

    def createModel(self, model_type, bow_corpus, dictionary, processed_corpus, number_topics):
        print(f"Training {model_type} model. This may take several minutes depending on the size of the corpus.")
        model = None
        iterations = self.settings['iterations'] if 'iterations' in self.settings else 50
        os.environ['MALLET_HOME'] = self.settings['mallet_path']
        mallet_path = os.path.join(self.settings['mallet_path'], 'bin', 'mallet')
        model = models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus, num_topics=number_topics, id2word=dictionary, iterations=iterations)
        coherence_model = self.getCoherenceModel(model, self.settings['coherence_measure'], bow_corpus, processed_corpus, dictionary)
        print(f"Model coherence score: {coherence_model.get_coherence()}")
        return model

    def getLDAVisPreparedData(self):
        lda_model = models.LdaModel(id2word=self.model.id2word, num_topics=self.model.num_topics, alpha=self.model.alpha, eta=0)
        lda_model.state.sstats[...] = self.model.wordtopics
        lda_model.sync_state()
        return pyLDAvis.gensim_models.prepare(lda_model, self.bow_corpus, self.dictionary)

class HDPModel(GensimModel):

    def createModel(self, model_type, bow_corpus, dictionary, processed_corpus, _):
        print(f"Training {model_type} model. This may take several minutes depending on the size of the corpus.")
        model = models.HdpModel(bow_corpus, dictionary)
        coherence_model = self.getCoherenceModel(model, self.settings['coherence_measure'], bow_corpus, processed_corpus, dictionary)
        print(f"Model coherence score: {coherence_model.get_coherence()}")
        return model
