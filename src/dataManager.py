import logging as log
import pandas as pd
from pathlib import Path
import pickle
import pymysql
import os

PROCESSED_DATA_FOLDER = "processed_data"
OUTPUT_FOLDER = "Output"


### GENERAL ###


def save_file(destination, content):
    """Pickle given structure to destination."""
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))
    with open(destination, 'wb') as out:
        pickle.dump(content, out)


def load_file(file_path):
    """Load pickled file from path."""
    if not os.path.isfile(file_path):
        print(f"{file_path} not found.")
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def getFilePath(settings, file, skip_node=False):
    datasetName = settings['datasetName']
    node = settings['node']
    return os.path.join(PROCESSED_DATA_FOLDER, f"{datasetName}_{node if not skip_node else ''}_{file}")


def getOutputDir(settings):
    """Return path to output directory while creating if needed."""
    output_dir = os.path.join(os.getcwd(), OUTPUT_FOLDER, settings['filters']['out']['folder'])
    checkDirectory(output_dir)
    return output_dir


def checkPathExists(destination):
    """Create directory tree if it does not exist."""
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))


def fileExists(file_path):
    """Return whether path exists and is a file."""
    return os.path.isfile(file_path)


def checkDirectory(output_dir):
    """Create directory if it does not exist."""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


### TEMP FILES ###


def save_tmp(settings, suffix, content):
    """Pickle given structure to temporary folder."""
    destination = getFilePath(settings, suffix, skip_node=True)
    save_file(destination, content)


def load_tmp(settings, suffix):
    """Load pickled file from temporary folder."""
    source = getFilePath(settings, suffix, skip_node=True)
    return load_file(source)


### DATAFRAMES ###

def save_df(settings, file, df):
    """Saves dataframe to temporary folder."""
    file_path = getFilePath(settings, file)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    df.to_pickle(file_path)


def load_df(settings, file):
    """Loads dataframe from temporary folder."""
    file_path = getFilePath(settings, file)
    if not os.path.isfile(file_path):
        print(f"{file_path} not found.")
        return None
    return pd.read_pickle(file_path)


### DATABASE ###


def get_connection(db_settings):
    connection = pymysql.connect(
        host=db_settings['host'],
        user=db_settings['user'],
        password=db_settings['password'],
        db=db_settings['db'],
        charset=db_settings['charset'] if 'charset' in db_settings else 'utf8',
        use_unicode=True,
    )
    return connection


def get_query(db_settings):
    query = ""
    filePath = os.path.join(os.getcwd(), db_settings['query'])
    if os.path.isfile(filePath):
        with open(filePath, 'r', encoding='utf-8') as f:
            query = f.read().strip('/r/n')
    else:
        query = db_settings['query']
    return query


def read_file(settings, dataFile):
    data_type = Path(dataFile).suffix
    encoding = settings['encoding'] if 'encoding' in settings else 'utf-8'
    selected_columns = [settings['corpusFieldName'], settings['dateFieldName']]
    if 'idFieldName' in settings:
        selected_columns.append(settings['idFieldName'])
    if data_type == '.json':
        json_orientation = settings['json_orientation'] if 'json_orientation' in settings else None
        df = pd.read_json(dataFile, orient=json_orientation, encoding=encoding)
        df = df[selected_columns]
    elif data_type == '.csv':
        df = pd.read_csv(dataFile, na_filter=False, usecols=selected_columns)
    elif data_type == '.xlsx':
        df = pd.read_excel(dataFile, na_filter=False, usecols=selected_columns)
    else:
        raise Exception
    total_items = df.shape[0]
    df.dropna(inplace=True)
    nan_items = total_items - df.shape[0]
    print(
        f"Loading {df.shape[0]} items from {dataFile} - {total_items} total items, {nan_items} NaN values were detected and removed.")
    return df


def read_data(settings):
    df = None
    dateFieldName = settings['dateFieldName']
    if 'verbose' in settings and settings['verbose']:
        # hack, use custom level
        log.basicConfig(format="%(message)s", level=log.INFO)
    if 'dataSource' in settings:
        dataSource = os.path.join(os.getcwd(), 'Data', settings['dataSource'])
    if 'db_settings' in settings:
        query = get_query(settings['db_settings'])
        db_connector = get_connection(settings['db_settings'])
        df = pd.read_sql(query, db_connector)
    elif os.path.isdir(dataSource):
        for filename in os.listdir(dataSource):
            file_df = read_file(settings, os.path.join(dataSource, filename))
            if df is None:
                df = file_df
            else:
                df = df.append(file_df)
    else:
        df = read_file(settings, dataSource)
    if 'lang' in settings:
        log.info(f"Filtering language. Only retaining {settings['lang'][1]} entries.")
        df = df[df[settings['lang'][0]] == settings['lang'][1]]
    log.info(f"Loaded {df.shape[0]} items.")
    log.info("Dataset preview:")
    log.info(df.head())
    log.info("Fields:")
    log.info(df.columns.values)
    log.basicConfig(format="%(message)s", level=log.ERROR)
    # Convert to datetime format (useful to filter by date) and round to nearest day
    if 'roundToDay' in settings and settings['roundToDay'] is True:
        df[dateFieldName] = pd.Series(pd.to_datetime(df[dateFieldName])).dt.round("D")
    else:
        df[dateFieldName] = pd.Series(pd.to_datetime(df[dateFieldName]))
    df = df.set_index([dateFieldName])
    df.sort_index(inplace=True)
    return df
