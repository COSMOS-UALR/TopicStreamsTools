import bitermplus
import numpy as np
import pickle as pkl
import pyLDAvis
from scipy import sparse
import tmplot as tmp

from ..dataManager import save_df
from .baseModel import BaseModel
from .output import saveToExcel, saveFigures, saveInteractivePage, saveModel
from .input import loadModel

from sklearn.feature_extraction.text import CountVectorizer

from ..dataManager import load_tmp, save_tmp, load_df

DOCS_VEC_FILE = 'docs_vec.obj'
DOCS_TO_REMOVE_FILE = 'docs_to_remove.obj'
VOC_FILE = 'vocabulary.obj'
CSR_DOC_FILE = 'csr_doc.obj'
BITERMS_FILE = 'biterms.obj'
DOC_WORD_FILE = 'doc_word.obj'
DOC_TOPIC_FILE = 'doc_topic.obj'

class BitermModel(BaseModel):

    def __init__(self, settings):
        super().__init__(settings)


    def loadProcessedData(self, settings):
        """Load multiple metadata files for topic modeling."""
        self.docs_vec = load_tmp(settings, DOCS_VEC_FILE)
        if self.settings['removeEmptyVecs']:
            self.docs_to_remove = load_tmp(settings, DOCS_TO_REMOVE_FILE)
        self.vocabulary = load_tmp(settings, VOC_FILE)
        self.csr_doc_word = load_tmp(settings, CSR_DOC_FILE)
        self.biterms = load_tmp(settings, BITERMS_FILE)
        self.doc_word_matrix = load_tmp(settings, DOC_WORD_FILE)


    def processData(self, corpus_df):
        """Process data from given corpus df."""
        settings = self.settings
        texts = corpus_df[settings['corpusFieldName']].values
        self.texts = texts
        vec = CountVectorizer(stop_words='english')
        # vec = CountVectorizer()
        self.doc_word_matrix = vec.fit_transform(texts).toarray()
        # Obtaining terms frequency in a sparse matrix and corpus vocabulary
        # self.doc_word_matrix, self.vocabulary, _ = bitermplus.get_words_freqs(texts)
        # tf = np.array(self.doc_word_matrix.sum(axis=0)).ravel()
        # Vectorizing documents
        self.vocabulary = np.array(vec.get_feature_names())
        self.docs_vec = bitermplus.get_vectorized_docs(texts, self.vocabulary)
        # If vetorized rows become empty, they need to be purged here in order for pyLDAVis to run.
        if settings['removeEmptyVecs']:
            self.docs_to_remove = [i for i, row in enumerate(self.docs_vec) if len(row) == 0]
            self.docs_vec = np.delete(self.docs_vec, self.docs_to_remove, 0).tolist()
            self.doc_word_matrix = np.delete(self.doc_word_matrix, self.docs_to_remove, 0).tolist()
        # Generating biterms
        self.biterms = bitermplus.get_biterms(self.docs_vec)
        self.csr_doc_word = sparse.csr_matrix(self.doc_word_matrix)
        # Dump processed data to files for faster loading
        if settings['removeEmptyVecs']:
            save_tmp(settings, DOCS_TO_REMOVE_FILE, self.docs_to_remove)
        save_tmp(settings, DOCS_VEC_FILE, self.docs_vec)
        save_tmp(settings, VOC_FILE, self.vocabulary)
        save_tmp(settings, CSR_DOC_FILE, self.csr_doc_word)
        save_tmp(settings, BITERMS_FILE, self.biterms)
        save_tmp(settings, DOC_WORD_FILE, self.doc_word_matrix)


    def train(self, corpus_df):
        """Will attempt to load the model and its associated data, will train and create if they cannot be found or retrainModel is True."""
        settings = self.settings
        try:
            self.model = loadModel(settings, self.MODEL_FILE + ".pkl")
            self.distributionDF = load_df(settings, self.DISTRIB_FILE)
            self.wordsDF = load_df(settings, self.WORDS_FILE)
            self.document_topics_matrix = load_tmp(settings, DOC_TOPIC_FILE)
        except FileNotFoundError as e:
            print(str(e))
            settings['retrainModel'] = True
        if settings['reloadData'] or settings['retrainModel']:
            self.model = bitermplus.BTM(self.csr_doc_word, self.vocabulary, T=self.settings['numberTopics'], M=20, alpha=50/8, beta=0.01, seed=12321)
            self.document_topics_matrix = self.model.fit_transform(self.docs_vec, self.biterms, iterations=20)
            save_tmp(settings, DOC_TOPIC_FILE, self.document_topics_matrix)
            def save_model_func(model, path):
                with open(path, "wb") as file:
                    pkl.dump(model, file)
            saveModel(settings, self.MODEL_FILE + ".pkl", self.model, save_model_func)
            theta = tmp.get_theta(self.model).transpose()
            if corpus_df is None:
                corpus_df = self.loadCorpus(reload=True)
            self.distributionDF = self.getThetaDF(self.settings, corpus_df, theta)
            save_df(settings, self.DISTRIB_FILE, self.distributionDF)
            self.wordsDF = self.getTopTopicWords(self.settings, self.model)


    def getThetaDF(self, settings, corpusDF, theta):
        """Return dataframes holding topic distributions for each document, with id or timestamped index."""
        if settings['removeEmptyVecs']:
            corpusDF = corpusDF.reset_index().drop(self.docs_to_remove)#.set_index(self.settings['dateFieldName'])
        else:
            corpusDF = corpusDF.reset_index()#.set_index(self.settings['dateFieldName'])
        timestamps = corpusDF[settings['dateFieldName']].values
        if 'idFieldName' in settings:
            ids = corpusDF[settings['idFieldName']].values
        topic_distrib_df = theta
        # Set IDs as index if defined then insert timestamp. If not, set timestamp as index
        topic_distrib_df = topic_distrib_df.set_index(ids if 'idFieldName' in settings else timestamps)
        if 'idFieldName' in settings and 'dateFieldName' in settings:
            topic_distrib_df.insert(0, settings['dateFieldName'], timestamps)
        save_df(settings, self.DISTRIB_FILE, topic_distrib_df)
        return topic_distrib_df


    def getTopTopicWords(self, settings, model):
        phi = tmp.get_phi(model)
        words_df = tmp.get_top_topic_words(phi, settings['numberWords']).T
        save_df(settings, self.WORDS_FILE, words_df)
        return words_df


    def output(self):
        """Will write to disk the results of the model's findings."""
        settings = self.settings
        # METRICS
        # perplexity = bitermplus.perplexity(self.model.matrix_topics_words_, self.document_topics_matrix, self.csr_doc_word, 8)
        # coherence = bitermplus.coherence(self.model.matrix_topics_words_, self.csr_doc_word, M=20)
        # or
        # perplexity = self.model.perplexity_
        # coherence = self.model.coherence_
        if 'folder' in settings['filters']['out']:
            nbFigures = settings['filters']['out']['nbFigures']
            settings['moving_average_size'] = settings['filters']['out']['moving_average_size']
            saveToExcel(settings, self.distributionDF, self.wordsDF)
            saveFigures(settings, self.distributionDF, self.wordsDF, n=nbFigures)
            if settings['removeEmptyVecs']:
                saveInteractivePage(settings, self.getLDAVisPreparedData())

    def getLDAVisPreparedData(self):
        # tmp.report(model=self.model, docs=self.texts)
        doc_lengths = np.count_nonzero(self.doc_word_matrix, axis=1)
        term_frequency = np.sum(self.doc_word_matrix, axis=0)
        topic_term_dists = tmp.get_phi(self.model).T.to_numpy()
        return pyLDAvis.prepare(topic_term_dists, self.document_topics_matrix, doc_lengths, self.vocabulary, term_frequency)
