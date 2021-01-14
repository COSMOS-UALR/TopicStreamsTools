from gensim import models
from tqdm import tqdm

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
    output_dir = os.path.join(os.getcwd(), 'Output', settings['datasetName'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir

### EXCEL ###

def save_to_excel(settings, distributionDF, wordsDF):
    print("Writing to excel. This may take a few minutes for larger corpora.")
    fileName = f"TopicDistribution_{settings['datasetName']}.xlsx"
    output_dir = save_to_output(settings)
    output_dest = os.path.join(output_dir, fileName)
    with pd.ExcelWriter(output_dest) as writer:
        wordsDF.to_excel(writer, index=True, header=True, sheet_name='Topic Words')
        distributionDF.to_excel(writer, index=True, header=True, sheet_name='Topic Distribution')
    print(f"Finished writing topic distribution data to {output_dest}.")


### FIGURES ###

def getColors():
    static_colors = [
        '#3498db',
        '#8e44ad',
        '#c0392b',
        '#76d7c4',
        '#e67e22',
        '#909497',
        '#212f3d',
        '#9c640c',
        '#28b463',
        '#21618c',
    ]
    for color in static_colors:
        yield color
    # Generate random colors once first 10 have been used
    while True:
        r = lambda: random.randint(0,255)
        hex_number = '#%02X%02X%02X' % (r(),r(),r())
        yield hex_number

def get_moving_average_window_size(settings, df):
    dataset_size = df.shape[0]
    percentile = settings['moving_average_size']
    window_size = round((dataset_size * percentile) / 100)
    print(f"\nApplying moving average window of {window_size} - {percentile}% of the dataset of size {dataset_size}")
    return window_size

def save_individual_plot(settings, dft, topic, window_size, color, output_dir):
    df = dft.iloc[topic]
    # Smooth curve
    if window_size > 0:
        df = df.rolling(window_size).mean()
    plot = df.plot(color=color)
    if 'x_label' in settings:
        plot.set_ylabel(settings['x_label'])
    if 'y_label' in settings:
        plot.set_ylabel(settings['y_label'])
    fig = plot.get_figure()
    fig.savefig(os.path.join(output_dir, f'Topic_{topic}_{settings["moving_average_size"]}MA.png'))
    fig.clf()

def save_overlapping_plot(settings, dft, topic_group, window_size, output_dir):
    colors = getColors()
    for topic in tqdm(topic_group):
        df = dft.iloc[topic]
        # Smooth curve
        if window_size > 0:
            df = df.rolling(window_size).mean()
        plot = df.plot(color=next(colors), label=f'Topic {topic}')
    if 'addLegend' in settings and settings['addLegend'] is True:
        box = plot.get_position()
        plot.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 1])
        plot.legend(bbox_to_anchor=(0.5, -0.170), loc='upper center', ncol=5)
    fig = plot.get_figure()
    fig.savefig(os.path.join(output_dir, f'Topics_{"-".join(str(x) for x in topic_group)}_{settings["moving_average_size"]}MA.png'))
    fig.clf()

def save_figures(settings, topics_df, words_df, n=5):
    selected_topics = words_df.head(n).index.values.tolist()
    dft = topics_df.transpose()
    output_dir = save_to_output(settings)
    window_size = get_moving_average_window_size(settings, topics_df)
    colors = getColors()
    print("Plotting figures...")
    for topic in tqdm(selected_topics):
        save_individual_plot(settings, dft, topic, window_size, next(colors), output_dir)
    # Multiple topics
    if 'topicGroups' in settings:
        topic_groups = settings['topicGroups']
        for topic_group in topic_groups:
            save_overlapping_plot(settings, dft, topic_group, window_size, output_dir)
    print(f"Finished plotting figures to {output_dir}.")
