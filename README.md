# TopicStreamsTools

This tool provides a quick way to generate topic streams for a given corpus. JSON and CSV files are accepted.

![Settings](/images/topicStreamExample.png)

For now, the quickest way to use the tool is to modify the settings in main.py.

```python
    settings = {
        'datasetName': 'COVID',            
        'filePath': 'KnownMisinfo.xlsx - Table1.csv',
        'corpusFieldName': 'title',
        'idFieldName': 'debunking_date',
        'reloadData': False,               # Will re-read input file and train a new model with the updated data
        # Advanced settings
        'numberTopics': 20,
        'numberWords': 10,
        'moving_avg_window_size': 20,
        'retrainModel': False,             # Will use the currently saved data and train a new model (useful to try different settings without processing the same corpus)
        'start_date': "2020-02-01",        # Optional - will only select items from this date when creating the topic distribution matrix
        'end_date': "2020-03-01",          # Optional - will only select items up to this date when creating the topic distribution matrix
        'minimumProbability': 0.00000001,
        'nbFigures': 5,
    }
```

## Instructions

1. Move your dataset (CSV or JSON) to the Data folder.
2. Make sure you have installed the requirements:
```python
    pip install -r requirements.txt
```
3. Open main.py in your preferred IDE and , referring to the documentation below, update your settings accordingly. The first four parameters must be filled, the rest can be left as is.
4. Execute main.py. You may need to wait up to an hour if running the tool for the first time on a large dataset (~1GB). Most sets should be done in minutes.
5. Open the Output/{datasetName} folder to find your results.

## Output

You can find the result of the analysis in the Output folder, in the subfolder you specified in the 'datasetName' parameter.
You will find a set of figures for the five most common topics, as well as an excel sheet containing the topic words and the total average topic distribution data for each day. Advanced users can use this data to plot graphs.
![Text Field](/images/sheetTab.png)
![Text Field](/images/topicDistribution.png)

## Settings info

1. datasetName
A recognizable name of your choice for your output.

2. filePath
The name of the dataset file you are using - not the path. The file must be in the Data folder.

3. corpusFieldName
The name of the field the tool will use to create a model for analysis. Typically, this will be the most verbose field of your dataset.
![Text Field](/images/textField.png)

4. idFieldName
The name of the field the tool will use to identify each entry. This should be a date field in your dataset such as a date of publication.
![ID Field](/images/idField.png)

5. reloadData
Make sure this is set to True is your data has changed. Otherwise the program will use stored information and the output will stay the same.

6. numberTopics
The total number of topics you would like your corpus to be divided into. Experiment with this. Ten or 20 is usually a good number but larger datasets may obtain good results with 50 topics.

7. numberWords
The total number of words you would like to represent each topic. For example, if numberWords = 10 this will be the result:
![Words](/images/wordCount.png)

8. moving_avg_window_size
The size of the moving average window used to smooth the generated figures. Experiment with different numbers if working with a large dataset, your window size may need to increase.

9. retrainModel
Use the same data but retrain the modal. This is useful if you are an advanced user making changes to the model settings such as minimumProbability.

10. start_date
While the model is trained on the entire corpus for performance, you may want to focus on a specific period when creating your topic distribution matrix. This will only select objects that start from this data in your specified id field.

11. end_date
Same as above for the end date.

12. minimumProbability
Topics with a probability lower than this threshold will be filtered out.

13. nbFigures
Number of figures generated to the output folder. Keep this number under the number of topics.

14. lang
Optional tuple where the first item is the name of the language column/attribute and the second is the language.