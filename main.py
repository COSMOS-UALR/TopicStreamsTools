from src.framework import Framework

legacy_config = "topic_model.conf.yml"
config = "framework.conf.yml"

def main(config):
    pipeline = Framework(config)
    pipeline.run()


main(legacy_config)
