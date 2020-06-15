
import pandas as pd
import os

from src.IO import load_df, save_to_excel, save_figures, DISTRIB_FILE, WORDS_FILE
from src.model import get_model
from src.matrix import createMatrix

def main():

    settings = {
        'datasetName': 'COVID',
        'filePath': 'KnownMisinfo.xlsx - Table1.csv',
        'corpusFieldName': 'title',
        'idFieldName': 'debunking_date',
        # Advanced settings
        'numberTopics': 20,
        'numberWords': 10,
        'moving_avg_window_size': 20,
        'reloadData': False,                # Will re-read your input file and train a new model with the updated data
        'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
        'minimumProbability': 0.00000001,
        'nbFigures': 5,
    }

    # settings = {
    #     'datasetName': 'MURI',
    #     'filePath': 'MURI blog data.json',
    #     'corpusFieldName': 'post',
    #     'idFieldName': 'date',
    #     'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    #     'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    #     'start_date': "2015-01-01",         # Optional - will only select items from this date when creating the topic distribution matrix
    #     'end_date': "2016-01-01",           # Optional - will only select items up to this date when creating the topic distribution matrix
    #     'numberTopics': 50,
    #     'numberWords': 10,
    #     'moving_avg_window_size': 50,
    #     'minimumProbability': 0.00000001,
    #     'nbFigures': 5,
    # }

    print('Attempting to load Dataframes...')
    distributionDF = load_df(settings['datasetName'], DISTRIB_FILE)
    wordsDF = load_df(settings['datasetName'], WORDS_FILE)

    if settings['reloadData'] or settings['retrainModel'] or (distributionDF is None or wordsDF is None):
        model, bow_corpus, ids = get_model(settings)
        print('Calculating Dataframes...')
        distributionDF, wordsDF = createMatrix(settings, model, bow_corpus, ids)

    save_to_excel(settings, distributionDF, wordsDF)

    save_figures(settings, distributionDF, wordsDF, n=settings['nbFigures'])

main()
