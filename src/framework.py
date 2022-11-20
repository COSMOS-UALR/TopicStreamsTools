from datetime import datetime
import yaml
from .topicModel.topicModel import TopicModelNode
from .engagementBehavior.engagementBehavior import EngagementBehaviorNode
from .commenterNetwork.commenterNetwork import CommenterNetwork


class Framework:

    def __init__(self, config) -> None:
        """Read configuration file and initialize nodes."""
        with open(config, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            self.nodes = []
            if 'settings' in cfg:
                self.handleLegacy(cfg)
            else:
                self.project_name = cfg['project']
                self.routing = cfg['routing']
                for node in self.routing['nodes']:
                    node_settings = node[list(node.keys())[0]]
                    self.nodes.append(self.nodeFactory(node_settings['node_type'])(self.project_name, node))

    def run(self):
        """Execute nodes."""
        out = {}
        for node in self.nodes:
            node_id = node.settings['node']
            node_type = node.settings['node_type']
            start_time = datetime.now()
            print(f"BEGIN {node_id} - {node_type} at {start_time}")
            out = node.run(out)
            end_time = datetime.now()
            delta = end_time - start_time
            print(f"END NODE {node} at {end_time}. Took {delta.seconds/86400:.0f}d {(delta.seconds % 86400) / 3600:.0f}h {(delta.seconds % 3600) / 60:.0f}m {delta.seconds % 60}s.")

    def nodeFactory(self, node_type):
        """Build node for each module type."""
        node_builder = {
            'topic_model': TopicModelNode,
            'channel_engagement': EngagementBehaviorNode,
            'comment_behavior': CommenterNetwork,
        }
        return node_builder.get(node_type)

    def handleLegacy(self, cfg):
        """Adapts old, topic-model-only architecture to the new configuration pattern"""
        settings = cfg['settings']
        project_name = settings['datasetName']
        settings['model'] = settings['model_type'] if 'model_type' in settings else 'LDA'
        settings['filters'] = {
            'in': {
                'file': True,
            },
            'out': {
                'folder': project_name,
                'nbFigures': settings['nbFigures'],
                'moving_average_size': settings['moving_average_size'],
            },
        }
        self.nodes.append(TopicModelNode(project_name, {'topic_model': settings}))
