domestic_extremism_comments_settings = {
    'datasetName': 'DE_comments',
    'dataSource': 'YT_Comments_November_1_2020_to_March_1_2021_DE.csv',
    'corpusFieldName': 'comment_displayed',
    'dateFieldName': 'published_date',
    'roundToDay': False,
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_average_size': 5,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
    'addLegend': True,
}

domestic_extremism_related_videos_settings = {
    'datasetName': 'DE_related_videos',
    'dataSource': 'YT_Related_Videos_November_1_2020_to_March_1_2021_DE.csv',
    'corpusFieldName': 'title',
    'dateFieldName': 'published_date',
    'roundToDay': False,
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_average_size': 5,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
    'addLegend': True,
}

domestic_extremism_videos_settings = {
    'datasetName': 'DE_videos',
    'dataSource': 'YT_Videos_November_1_2020_to_March_1_2021_DE.csv',
    'corpusFieldName': 'video_title',
    'dateFieldName': 'published_date',
    'roundToDay': False,
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_average_size': 5,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
    'addLegend': True,
}

domestic_extremism_twitter_settings = {
    'datasetName': 'DE_twitter',
    'dataSource': 'Twitter_Posts_November_1_2020_to_March_1_2021_DE.csv',
    'corpusFieldName': 'text',
    'dateFieldName': 'created_at',
    'roundToDay': False,
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_average_size': 5,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
    'addLegend': True,
}

domestic_extremism_parler_settings = {
    'datasetName': 'DE_parler',
    'dataSource': 'parler_posts2.csv',
    'corpusFieldName': 'body',
    'dateFieldName': 'created_at',
    'roundToDay': False,
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_average_size': 5,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
    'addLegend': True,
}

covid_yt_settings = {
    'datasetName': 'COVID_March_YT_Comments',
    'dataSource': 'COVID_keywords_March_comments.json',
    'corpusFieldName': 'comment_displayed',
    'dateFieldName': 'published_date',
    'json_orientation': 'columns',
    'roundToDay': False,
    'model_type': 'LDA',
    'start_date': "2020-03-01",
    'end_date': "2020-04-01",
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_average_size': 3,
    # 'moving_avg_window_size': 1000,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'distributionInWorksheet': False,
    'nbFigures': 0,
    'topicGroups': [[7, 17]],
    'addLegend': True,
}


covid_videos_settings = {
    'datasetName': 'COVID_March_videos',
    'dataSource': 'COVID_March_videos.json',
    'corpusFieldName': 'video_title',
    'dateFieldName': 'published_date',
    'json_orientation': 'columns',
    'roundToDay': False,
    'model_type': 'LDA',
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_average_size': 15,
    # 'moving_avg_window_size': 1000,
    'reloadData': True,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': True,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'distributionInWorksheet': True,
    'nbFigures': 10,
    'topicGroups': [[0,17,6]],
    'addLegend': True
}

jan_yt_settings = {
    'datasetName': 'January_comments',
    'dataSource': 'January_2020.json',
    'corpusFieldName': 'comment_displayed',
    'dateFieldName': 'published_date',
    'json_orientation': 'columns',
    'roundToDay': False,
    'model_type': 'LDA',
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_average_size': 1,
    # 'moving_avg_window_size': 1000,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': True,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
}

posts_settings = {
    'datasetName': 'posts',
    'dataSource': 'POST.json',
    'corpusFieldName': 'post',
    'dateFieldName': 'date',
    'idFieldName': 'blogpost_id',
    'json_orientation': 'records',
    'distributionInWorksheet': True,
    'roundToDay': False,
    # Advanced settings
    'numberTopics': 10,
    'numberWords': 10,
    'moving_average_size': 2,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
}

DOD_AUS_settings = {
    'datasetName': 'DOD_AUSI_JSON',
    'dataSource': 'DOD_AUSI_JSON.json',
    'corpusFieldName': 'content',
    'dateFieldName': 'published_date',
    'json_orientation': 'records',
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_average_size': 2,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 10,
    'topicGroups': [[2, 7, 6, 1, 5]],
}


misc_sets = {
    'datasetName': 'descrp',
    'dataSource': 'descrp.json',
    'corpusFieldName': 'description',
    'dateFieldName': 'published_date',
    'model_type': 'LDA',
    'json_orientation': 'records',
    'encoding': 'iso8859_14',
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_average_size': 2,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 0,
}

covid_settings = {
    # 'datasetName': 'COVID_titles',
    'datasetName': 'COVID_themes',
    'dataSource': 'covid.csv',
    # 'corpusFieldName': 'title',
    'corpusFieldName': 'THEME',
    'dateFieldName': 'debunking_date',
    'roundToDay': True,
    'model_type': 'LDA',
    # Advanced settings
    'numberTopics': 20,
    'numberWords': 10,
    'moving_average_size': 2,
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'minimumProbability': 0.00000001,
    'nbFigures': 20,
    # 'topicGroups': [[10,12,17,18]],
    'topicGroups': [[2, 13, 15]],
    'minimumProbability': 0.00000001,
    'distributionInWorksheet': False,
    'addLegend': True,
}

muri_settings = {
    'datasetName': 'MURI',
    'dataSource': 'MURI blog data.json',
    'corpusFieldName': 'post',
    'dateFieldName': 'date',
    'reloadData': False,                # Will re-read your input file and train a new model with the updated data
    'retrainModel': False,              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    'start_date': "2015-01-01",         # Optional - will only select items from this date when creating the topic distribution matrix
    'end_date': "2016-01-01",           # Optional - will only select items up to this date when creating the topic distribution matrix
    'numberTopics': 50,
    'numberWords': 10,
    'moving_average_size': 5,
    'minimumProbability': 0.00000001,
    'nbFigures': 5,
    'lang': ('language', 'English'),   # Optional - if specified, will filter by column 'language' whose value is 'English'
}
