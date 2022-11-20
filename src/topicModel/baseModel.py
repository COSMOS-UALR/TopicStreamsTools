from ..dataManager import load_tmp, read_data, get_connection, fetchData
from .input import queryChannelData

# Topic distribution a document must meet to belong to a topic. Used to communicate what documents to keep from one node to another if fileterd by topic.
BELONGING_THRESHOLD = 0.3

class BaseModel:

    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.IDS_FILE = 'ids.pkl'
        self.belonging_threshold = BELONGING_THRESHOLD
        self.MODEL_FILE = f"{self.settings['model']}_model"
        self.DISTRIB_FILE = 'distribDataframe.pkl'
        self.WORDS_FILE = 'wordsDataframe.pkl'
        self.distributionDF = None
        self.wordsDF = None


    def loadCorpus(self, reload=False):
        """Will fetch raw data, either from IDs given by previous node, form disk, or by reading the source."""
        corpus_df = None
        if 'node' in self.settings['filters']['in']:
            self.settings['verbose'] = False
            filtered_ids = []
            corpus_df = read_data(self.settings)
            filtered_ids = load_tmp(self.settings, self.IDS_FILE)
            corpus_df = corpus_df[corpus_df[self.settings['idFieldName']].isin(filtered_ids)]
            print(f"Kept {corpus_df.shape[0]} items.")
        elif self.settings['reloadData'] or reload:
            if 'db_settings' in self.settings['filters']['in']:
                channel_ids = self.settings['filters']['in']['channel_ids'] if 'channel_ids' in self.settings['filters']['in'] else None
                if 'query' in self.settings['filters']['in']['db_settings']:
                    db_connector = get_connection(self.settings['filters']['in']['db_settings'])
                    CHUNKSIZE = 10000
                    table = 'posts'
                    query = self.settings['filters']['in']['db_settings']['query']
                    corpus_df = fetchData(db_connector, query, table, chunksize=CHUNKSIZE)
                else:
                    corpus_df = queryChannelData(self.settings, channel_ids)
            else:
                corpus_df = read_data(self.settings)
        return corpus_df


    def getData(self, corpus_df):
        """Will attempt to load processed file, or regenerate if they are missing or reloadData is True."""
        settings = self.settings
        settings['verbose'] = True
        if not settings['reloadData']:
            try:
                self.loadProcessedData(settings)
            except FileNotFoundError as e:
                print(str(e))
                self.settings['reloadData'] = True
                corpus_df = self.loadCorpus(reload=True)
        if self.settings['reloadData']:
            self.processData(corpus_df)


    def loadProcessedData():
        raise NotImplementedError


    def processData():
        raise NotImplementedError


    def train():
        raise NotImplementedError


    def output():
        raise NotImplementedError
