from .gensimModels import LDAModel, LDAMalletModel, HDPModel
from .biterm import BitermModel

class TopicModelNode:

    def __init__(self, project_name, settings) -> None:
        node = list(settings.keys())[0]
        self.settings = settings[node]
        self.settings['node'] = node
        self.settings['datasetName'] = project_name
        self.settings['model'] = settings[node]['model_type']
        if 'dataSource' in self.settings:
            self.settings['dataSource'] = self.settings['dataSource']
        elif 'file' in self.settings:
            self.settings['dataSource'] = self.settings['file']


    def run(self, previous_node_output):
        model = self.getModel()
        corpus_df = model.loadCorpus()
        model.getData(corpus_df)
        model.train(corpus_df)
        out = model.output()
        return out


    def getModel(self):
        settings = self.settings
        models = {
            'LDA': LDAModel(settings),
            'LDA-Mallet': LDAMalletModel(settings),
            'HDP': HDPModel(settings),
            'Biterm': BitermModel(settings),
        }
        if settings['model_type'] not in models:
            raise Exception(msg='Invalid model.')
        return models[settings['model_type']]
