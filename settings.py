comments_settings = {
    'datasetName': 'comments',
    'dataSource': 'VideoCommentsWithChannelandCategoryJan1-April302020.json',
    'corpusFieldName': 'comment_displayed',
    'idFieldName': 'published_date',
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'numberTopics': 10,
    'numberWords': 10,
    'moving_avg_window_size': 200,
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
}

yt_settings = {
    'datasetName': 'February_comments',
    'dataSource': 'February_2020.json',
    'corpusFieldName': 'comment_displayed',
    'idFieldName': 'published_date',
    'json_orientation': 'columns',
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_avg_window_size': 100,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
    # 'topicGroups': [[2, 7, 6, 1, 5]],
}

DOD_AUS_settings = {
    'datasetName': 'DOD_AUSI_JSON',
    'dataSource': 'DOD_AUSI_JSON.json',
    'corpusFieldName': 'content',
    'idFieldName': 'published_date',
    'json_orientation': 'records',
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_avg_window_size': 1000,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
    'topicGroups': [[2, 7, 6, 1, 5]],
}

covid_settings = {
    'datasetName': 'test',
    # 'datasetName': 'COVID_themes',
    'dataSource': 'covid.csv',
    'corpusFieldName': 'title',
    # 'corpusFieldName': 'THEME',
    'idFieldName': 'debunking_date',
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_avg_window_size': 20,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
    # 'topicGroups': [[10,12,17,18]],
    'topicGroups': [[3,9,16]],
}

muri_settings = {
    'datasetName': 'MURI',
    'dataSource': 'MURI blog data.json',
    'corpusFieldName': 'post',
    'idFieldName': 'date',
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'start_date': "2015-01-01",         # Optional - will only select items from this date when creating the topic distribution matrix
    'end_date': "2016-01-01",           # Optional - will only select items up to this date when creating the topic distribution matrix
    'numberTopics': 50,
    'numberWords': 10,
    'moving_avg_window_size': 50,
    'minimumProbability': 0.00000001,
    'nbFigures': 5,
    'lang': ('language', 'English'),   # Optional - if specified, will filter by column 'language' whose value is 'English'
}
