import yaml

from .topicModel.topicModel import TopicModelNode

class Framework:

    def __init__(self, config) -> None:
        with open(config, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            self.project_name = cfg['project']
            self.routing = cfg['routing']
            self.nodes = self.routing['nodes']
            self.node_1 = TopicModelNode(self.project_name, self.nodes[0])
            self.node_2 = TopicModelNode(self.project_name, self.nodes[1])

    def run(self):
        # self.node_1.run()
        self.node_2.run()
        return
