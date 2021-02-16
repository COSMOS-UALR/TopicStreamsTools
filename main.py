
import pandas as pd
import os

from src.IO import load_df, save_to_excel, save_figures, DISTRIB_FILE, WORDS_FILE
from src.model import get_model
from src.matrix import createMatrix
from settings import *

settings = covid_yt_settings

def main(settings):

    print('Attempting to load Dataframes...')
    distributionDF = load_df(settings, DISTRIB_FILE)
    wordsDF = load_df(settings, WORDS_FILE)

    if settings['reloadData'] or settings['retrainModel'] or (distributionDF is None or wordsDF is None):
        model_type = settings['model_type'] if 'model_type' in settings else 'LDA'
        model, bow_corpus, ids = get_model(settings, model_type)
        print('Calculating Dataframes...')
        distributionDF, wordsDF = createMatrix(settings, model, bow_corpus, ids)

    save_to_excel(settings, distributionDF, wordsDF)
    
    save_figures(settings, distributionDF, wordsDF, n=settings['nbFigures'])

main(settings)
