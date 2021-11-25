from ..dataManager import load_tmp, read_data
from .input import loadData

# Topic distribution a document must meet to belong to a topic. Used to communicate what documents to keep from one node to another if fileterd by topic.
BELONGING_THRESHOLD = 0.3

class BaseModel:

    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.IDS_FILE = 'ids.pkl'
        self.belonging_threshold = BELONGING_THRESHOLD
        self.MODEL_FILE = f"{self.settings['model']}_model"


    def loadData(self):
        """Will fetch raw data, either from IDs given by previous node, form disk, or by reading the source."""
        if 'node' in self.settings['filters']['in']:
            self.settings['verbose'] = False
            filtered_ids = []
            corpus_df = read_data(self.settings)
            filtered_ids = load_tmp(self.settings, self.IDS_FILE)
            self.corpus_df = corpus_df[corpus_df[self.settings['idFieldName']].isin(filtered_ids)]
            print(f"Kept {corpus_df.shape[0]} items.")
        elif self.settings['reloadData']:
            self.corpus_df = read_data(self.settings)
        else:
            self.corpus_df = loadData(self.settings)


    def processData():
        raise NotImplementedError


    def train():
        raise NotImplementedError


    def output():
        raise NotImplementedError
