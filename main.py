
import pandas as pd
import os

from src.IO import load_df, save_to_excel, save_figures, DISTRIB_FILE, WORDS_FILE
from src.model import get_model
from src.matrix import createMatrix
from settings import *

settings = yt_settings

def main(settings):

    print('Attempting to load Dataframes...')
    distributionDF = load_df(settings['datasetName'], DISTRIB_FILE)
    wordsDF = load_df(settings['datasetName'], WORDS_FILE)

    if settings['reloadData'] or settings['retrainModel'] or (distributionDF is None or wordsDF is None):
        model, bow_corpus, ids = get_model(settings)
        print('Calculating Dataframes...')
        distributionDF, wordsDF = createMatrix(settings, model, bow_corpus, ids)

    save_to_excel(settings, distributionDF, wordsDF)
    
    save_figures(settings, distributionDF, wordsDF, n=settings['nbFigures'])

main(settings)
