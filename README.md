# TopicStreamsTools

This tool provides a quick way to generate topic streams for a given corpus.
JSON, Excel, and CSV files are accepted. However it is recommended to avoid CSV to limit ambiguous separators on large text fields. If the script is failing with a CSV file try with a JSON version.

See the latest changes [here](CHANGELOG.md).

![Settings](/images/topicStreamExample.png)


## Quick Start

To run the script with the sample dataset:
1. Install the script's requirements - it is recommended to do so in a virtual environment with Python 3.8.
```python
    pip3 install -r requirements.txt
```
2. Execute main.py. You may need to wait up to an hour if running the tool for the first time on a large dataset (~1GB). Most sets should be done in minutes. If you encounter errors, check first that the encoding of your file is correct and you have specified the correct JSON orientation if using.
3. Open the Output/LDA/{datasetName} folder to find your results.

## Using custom datasets

1. Move dataset to the Data folder.
    * This can be a single file, or a folder with multiple files with identical format.
2. Update the `topic_model.conf.yml` file according to your dataset and needs, detailed description [below](#settings-info). Typically, the first four parameters must be updated with each different dataset. The rest can be left as is and will run.

```yaml
settings:
    datasetName: COVID_Misinfo
    dataSource: KnownMisinfo.xlsx - Table1.csv
    corpusFieldName: title
    dateFieldName: debunking_date
    # Advanced settings
    model_type: LDA
    optimize_model: True
    multithreading: True
    roundToDay: False
    numberTopics: 20
    numberWords: 10
    nbFigures: 5
    moving_average_size: 5
    reloadData: False                # Will re-read your input file and train a new model with the updated data
    retrainModel: False              # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
    minimumProbability: 0.00000001
    # topicGroups: [[2, 13, 15]]
    minimumProbability: 0.00000001
    distributionInWorksheet: False
    addLegend: True
    db_settings:
            host: <server_ip>
            user: <your_login>
            password: <your_password>
            db: <your_database>
            query: query.sql
            encoding: utf-8
```

## Output

You can find the result of the analysis in the Output folder, in the subfolder you specified in the 'datasetName' parameter.
You will find a set of figures for the five most common topics, as well as an excel sheet containing the topic words and the total average topic distribution data for each day. Advanced users can use this data to plot graphs.

![Text Field](/images/sheetTab.png)
![Text Field](/images/topicDistribution.png)

## Settings info

* **datasetName**
A recognizable name of your choice for your output.

* **dataSource** (optional if using db_settings)
The name of the dataset file or folder to be used - not the path. The file or folder must be in the Data folder. If a folder is used, the tool will attempt to aggregate all the files in that folder. All files must use the same column format.

* **corpusFieldName**
The name of the field the tool will use to create a model for analysis. Typically, this will be the most verbose field of your dataset.

![Text Field](/images/textField.png)

* **dateFieldName** (formerly idFieldName)
The name of the field the tool will use to identify each entry. This should be a date field in your dataset such as a date of publication.

![Date Field](/images/idField.png)

* **model_type**
Type of topic model to use. Options: LDA, LDA-Mallet, HDP. Default is LDA (Recommended). With LDA, the model attempts to fit the corpus to the specified number of topics. HDP will attempt a best guess at the number of topics (with a maximum threshold of 150) and show  the 'numberTopics' most represented topics.

* **mallet_path**
If the topic model is LDA-Mallet, you will need to provide the path to its home directory. You can download mallet [here](http://mallet.cs.umass.edu/download.php).

* **optimize_model**
Setting this parameter to True will train models using a range of topic numbers and select the one with the highest coherence. This will increase the length of the first run time.

* **multithreading**
Will attempt to use multithreading to accelerate training. Known to cause issues when generating plots, you may need to rerun the script after the model has been saved.

* **coherence_measure**
The type of measure to give coherence score. Accepted values are `c_v`(default) and `u_mass`. Details on coherence measures can be found [here](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0).

* **roundToDay**
If defined and True, rounds dates to the nearest day. Better for larger corpora spanning over multiple months.

* **numberWords**
The total number of words you would like to represent each topic. For example, if numberWords = 10 this will be the result:
![Words](/images/wordCount.png)

* **nbFigures**
Number of figures generated to the output folder. Keep this number under the number of topics.

* **reloadData**
Make sure this is set to True is your data has changed. Otherwise the program will use stored information and the output will stay the same.

* **retrainModel**
Use the same data but retrain the modal. This is useful if you are an advanced user making changes to the model settings such as minimumProbability.

* **minimumProbability**
Topics with a probability lower than this threshold will be filtered out.

* **moving_average_size**
Percentile value, indicates what percentage of the dataset size to use as a moving average window to smooth the resulting figures. Values between 1 and 15 are recommended but you may need to experiment with different numbers. Note that larger values may make figures lose resolution. Depending on the composition of your dataset some dates may get lost if the value is too high. Set to 0 to forego using a moving average.

### Optional Settings

* **numberTopics**
If not using optimize_model, the total number of topics you would like your corpus to be divided into. Experiment with this. Ten or 20 is usually a good number but larger datasets may obtain good results with 50 topics.

* (JSON only) **json_orientation**
Specifies the orientation of the json file. Needed for JSON files. See the documentation below (arguments for 'orient') for acceptable values:
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html

* **db_settings**
If provided, this dictionary will override other data input settings and fetch data from the provided host. Adjust host address, login credentials, database, and encoding (optional, default = utf-8) accordingly. The query parameter can be either the name of a query file (txt or sql) located in the root of the project, or a simple string query as-is.

* **encoding**
Specifies the encoding of the file. Optional, UTF-8 is the default.

* **idFieldName**
Individual identifier such as a blog post ID, video ID, etc. If distributionInWorksheet is also set to true, this will generate an additional excel sheet in the form of an edge list. This is useful for analysts wishing to filter by individual topics to match particular items to a topic for further study.

![ID Field](/images/edgeList.png)

* **start_date**
While the model is trained on the entire corpus for performance, you may want to focus on a specific period when creating your topic distribution matrix. This will only select objects that start from this data in your specified id field. Example value: '2020-02-01'.

* **end_date**
Same as above for the end date. Example value: '2020-03-01'.

* **iterations**
Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.

* **passes**
Number of passes through the corpus during training.

* **lang**
Optional tuple where the first item is the name of the language field and the second is the language to keep.

* **topicGroups**
Optional list of topics to print on one graph. First run on your dataset then check what resulting figures you would like to group up.

* **distributionInWorksheet**
If True, will append an excel sheet with the raw distribution data for each topic. If idFieldName is also defined, will append an additional sheet with id fields matched to topics.

* **addLegend**
Add a legend to any overlapping plot.

* **x_label**
Add label to the x axis.

* **y_label**
Add label to the y axis.

* **alternate_legend**
Set to True to display plot legends on the right.


## Some useful references:

* [pyLDAvis: Topic Modelling Exploration Tool That Every NLP Data Scientist Should Know](https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know)
