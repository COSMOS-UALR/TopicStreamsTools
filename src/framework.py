import yaml

from .topicModel.topicModel import TopicModelNode
from .engagementBehavior.engagementBehavior import EngagementBehaviorNode


class Framework:

    def __init__(self, config) -> None:
        with open(config, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            self.project_name = cfg['project']
            self.routing = cfg['routing']
            self.nodes = []
            for node in self.routing['nodes']:
                node_settings = node[list(node.keys())[0]]
                self.nodes.append(self.nodeFactory(node_settings['node_type'])(self.project_name, node))

    def run(self):
        # for node in self.nodes:
        #     node.run()
        self.nodes[2].run()

    def nodeFactory(self, node_type):
        node_builder = {
            'topic_model': TopicModelNode,
            'channel_engagement': EngagementBehaviorNode,
        }
        return node_builder.get(node_type)
