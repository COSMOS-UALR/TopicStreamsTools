from src.framework import Framework


def main():
    config = "config.yml"
    pipeline = Framework(config)
    pipeline.run()


main()
