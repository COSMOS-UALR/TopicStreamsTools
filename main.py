from multiprocessing import freeze_support
from src.framework import Framework

legacy_config = "topic_model.conf.yml"
config = "framework.conf.yml"

def main(config):
    pipeline = Framework(config)
    pipeline.run()


if __name__ == "__main__":
    freeze_support()
    main(config)
