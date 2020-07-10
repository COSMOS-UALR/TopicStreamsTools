from gensim import models

import pickle, os, random

import pandas as pd

## GLOBALS ###

PROCESSED_DATA_FOLDER = "processed_data"
BOW_FILE = 'BOW.obj'
DICT_FILE = 'dictionary.obj'
ID_FILE = 'ids.obj'
DISTRIB_FILE = 'distribDataframe.pkl'
WORDS_FILE = 'wordsDataframe.pkl'

### GENERAL ###

def save_file(destination, content):
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
    with open(destination, 'wb') as out:
        pickle.dump(content, out)

def load_file(file_path):
    if not os.path.isfile(file_path):
        print(f"{file_path} not found.")
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)

### TEMP FILES ###

def save_tmp(datasetName, suffix, content):
    destination = os.path.join(PROCESSED_DATA_FOLDER, datasetName + '_' + suffix)
    save_file(destination, content)

def load_tmp(datasetName, suffix):
    source = os.path.join(PROCESSED_DATA_FOLDER, datasetName + '_' + suffix)
    return load_file(source)

### DATAFRAMES ###

def save_df(datasetName, file, df):
    file_path = os.path.join(PROCESSED_DATA_FOLDER, datasetName + '_' + file)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    df.to_pickle(file_path)

def load_df(datasetName, file):
    file_path = os.path.join(PROCESSED_DATA_FOLDER, datasetName + '_' + file)
    if not os.path.isfile(file_path):
        print(f"{file_path} not found.")
        return None
    return pd.read_pickle(file_path)

### MODELS ###

def getModelLocation(datasetName):
    return os.path.join(os.getcwd(), 'models', datasetName, 'LDAmodel')

def save_model(datasetName, model):
    destination = getModelLocation(datasetName)
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
    model.save(destination)

def load_model(datasetName):
    file_path = getModelLocation(datasetName)
    if not os.path.isfile(file_path):
        print(f"{file_path} not found.")
        return None
    return models.LdaModel.load(file_path)

### MULTIPLE FILES ###

def loadFiles(datasetName):
    model = load_model(datasetName)
    bow_corpus = load_tmp(datasetName, BOW_FILE)
    ids = load_tmp(datasetName, ID_FILE)
    return model, bow_corpus, ids

def loadData(datasetName):
    bow_corpus = load_tmp(datasetName, BOW_FILE)
    dictionary = load_tmp(datasetName, DICT_FILE)
    ids = load_tmp(datasetName, ID_FILE)
    return bow_corpus, dictionary, ids

### OUTPUT ###

def save_to_output(settings):
    output_dir = os.path.join('Output', settings['datasetName'])
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))
    return output_dir

### EXCEL ###

def save_to_excel(settings, distributionDF, wordsDF):
    fileName = 'TopicDistribution.xlsx'
    output_dir = save_to_output(settings)
    output_dest = output_dir + fileName
    with pd.ExcelWriter(output_dest) as writer:
        wordsDF.to_excel(writer, index=True, header=True, sheet_name='Topic Words')
        distributionDF.to_excel(writer, index=True, header=True, sheet_name='Topic Distribution')
    print(f"Finished writing topic distribution data to {output_dest}.")


### FIGURES ###

def getColor():
    r = lambda: random.randint(0,255)
    hex_number = '#%02X%02X%02X' % (r(),r(),r())
    return hex_number

def save_figures(settings, topics_df, words_df, n=5):
    selected_topics = words_df.head(n).index.values.tolist()
    dft = topics_df.transpose()
    output_dir = save_to_output(settings)
    for topic in selected_topics:
        df = dft.loc[topic]
        # Smooth curve
        df = df.rolling(settings['moving_avg_window_size']).mean()
        plot = df.plot(color=getColor())
        fig = plot.get_figure()
        fig.savefig(os.path.join(output_dir, f'Topic_{topic}.png'))
        fig.clf()
    print(f"Finished plotting figures to {output_dir}.")
