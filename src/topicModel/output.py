import matplotlib.pyplot as plt 
import os
import pandas as pd
import random
from tqdm import tqdm
import plotly.express as px
import pyLDAvis
import pyLDAvis.gensim

from ..dataManager import checkPathExists, getFilePath, getOutputDir


def saveModel(settings, file, model):
    file_path = getFilePath(settings, file)
    checkPathExists(file_path)
    model.save(file_path)


### Nodes ###


def getIDsInTopics(df, topics, belonging_threshold):
    """Return IDs of documents whose topic probability is over the threshold for given topic(s)."""
    query = ' | '.join([f'@df[{topic}] >= {belonging_threshold}' for topic in topics])
    df = df.query(query)
    return df.index.values


### EXCEL ###


def getEdgeListDF(settings, distributionDF):
    df = distributionDF.reset_index()
    df = df.melt(id_vars=['index', settings['dateFieldName']], var_name='Topic', value_name='Topic Distribution')
    df.sort_values(['index', 'Topic'], inplace=True)
    return df.rename(columns={"index": settings['idFieldName']})


def saveWorksheet(settings, writer, worksheet, df, dir, write_index):
    # Write to CSV if data is too large. Max excel sheet size is: 1048576, 16384
    if df.shape[0] > 1048576:
        fileName = f"{settings['model']}_{worksheet}_{settings['datasetName']}.csv"
        path = os.path.join(dir, fileName)
        df.to_csv(path, index=write_index, header=True)
        print(f"The {worksheet} data was too large to write as an Excel worksheet and was written to {path} instead.")
    else:
        df.to_excel(writer, index=write_index, header=True, sheet_name=worksheet)


def saveToExcel(settings, distributionDF, wordsDF):
    print("Writing to excel. This may take a few minutes for larger corpora.")
    fileName = f"{settings['model']}_TopicDistribution_{settings['datasetName']}.xlsx"
    output_dir = getOutputDir(settings)
    output_dest = os.path.join(output_dir, fileName)
    with pd.ExcelWriter(output_dest) as writer:
        wordsDF.to_excel(writer, index=True, header=True, sheet_name='Topic Words')
        if 'distributionInWorksheet' in settings and settings['distributionInWorksheet']:
            saveWorksheet(settings, writer, 'Topic Distribution', distributionDF, output_dir, True)
            if 'idFieldName' in settings:
                edgeListDF = getEdgeListDF(settings, distributionDF)
                saveWorksheet(settings, writer, 'Edge List', edgeListDF, output_dir, False)
    print(f"Wrote topic distribution data to {output_dest}.")


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
        r = lambda: random.randint(0, 255)
        hex_number = '#%02X%02X%02X' % (r(), r(), r())
        yield hex_number


def getMovingAverageWindowSize(settings, df):
    dataset_size = df.shape[0]
    percentile = settings['moving_average_size']
    window_size = round((dataset_size * percentile) / 100)
    print(f"Applying moving average window of {window_size} - {percentile}% of the dataset of size {dataset_size}")
    return window_size


def saveIndividualPlot(settings, dft, topic, window_size, color, output_dir):
    df = dft.iloc[topic]
    # Smooth curve
    if window_size > 0:
        df = df.rolling(window_size).mean()
    plot = df.plot(color=color)
    setMatPlotLibOptions(settings, plot)
    fig = plot.get_figure()
    fig.savefig(os.path.join(output_dir, f'{settings["model"]}_Topic_{topic}_{settings["moving_average_size"]}MA.png'))
    fig.clf()


def saveOverlappingPlot(settings, dft, topic_group, window_size, output_dir, filename=None):
    if filename is None:
        filename = f'{settings["model"]}_Topics_{"-".join(str(x) for x in topic_group)}_{settings["moving_average_size"]}MA'
    colors = getColors()
    plotly_df = pd.DataFrame(dft.transpose())
    topics_to_ommit = list(set(plotly_df.columns) - set(topic_group))
    plotly_df = plotly_df.drop(topics_to_ommit, axis=1)
    for topic in tqdm(topic_group, desc=f"Plotting topic group {topic_group}"):
        df = dft.iloc[topic]
        # Smooth curve
        if window_size > 0:
            df = df.rolling(window_size).mean()
            plotly_df[topic] = plotly_df[topic].rolling(window_size).mean()
        plot = df.plot(color=next(colors), label=f'Topic {topic}')
    # Drop NA values fort sorting
    plotly_df = plotly_df.dropna(axis=0, how='all')
    # Order columns so the legend displays labels in order of appearance
    plotly_df = plotly_df[plotly_df.columns[plotly_df.iloc[0].argsort()][::-1]]
    plotly_fig = px.line(plotly_df.reset_index(), x=plotly_df.index, y=plotly_df.columns)
    setPlotlyOptions(settings, plotly_fig)
    setMatPlotLibOptions(settings, plot, multiple_lines = True)
    plotly_fig.write_html(os.path.join(output_dir, f'{filename}.html'))
    fig = plot.get_figure()
    fig.savefig(os.path.join(output_dir, f'{filename}.png'))
    fig.clf()


def setPlotlyOptions(settings, plotly_fig):
    legend_on_right = settings['alternate_legend'] if 'alternate_legend' in settings else False
    plotly_fig.update_xaxes(tickformat="%m-%d-%Y")
    if legend_on_right:
        plotly_fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0.25))
    plotly_fig.update_layout(
        xaxis_title = settings['x_label'] if 'x_label' in settings else "",
        yaxis_title = settings['y_label'] if 'y_label' in settings else "Topic's probability distribution",
        legend_title = "Topics")


def setMatPlotLibOptions(settings, plot, multiple_lines = False):
    legend_on_right = settings['alternate_legend'] if 'alternate_legend' in settings else False
    if multiple_lines and 'addLegend' in settings and settings['addLegend'] is True:
        box = plot.get_position()
        if legend_on_right:
            plot.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plot.legend(bbox_to_anchor=(1, 1), ncol=1)
        else:
            plot.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 1])
            plot.legend(bbox_to_anchor=(0.5, -0.215), loc='upper center', ncol=5)
    if 'x_label' in settings:
        plot.set_ylabel(settings['x_label'])
    if 'y_label' in settings:
        plot.set_ylabel(settings['y_label'])


def saveFigures(settings, topics_df, words_df, n=5):
    selected_topics = words_df.head(n).index.values.tolist()
    # Set dates as index for plotting if idFieldName was used
    if 'idFieldName' in settings:
        topics_df = topics_df.reset_index(drop=True).set_index(settings['dateFieldName'])
    dft = topics_df.transpose()
    output_dir = getOutputDir(settings)
    window_size = getMovingAverageWindowSize(settings, topics_df)
    colors = getColors()
    for topic in tqdm(selected_topics, desc='Plotting individual charts'):
        saveIndividualPlot(settings, dft, topic, window_size, next(colors), output_dir)
    # Multiple topics
    # Save top topics by default
    top_topics_filename = f'Top_{n}_Topics_{settings["moving_average_size"]}MA'
    saveOverlappingPlot(settings, dft, selected_topics, window_size, output_dir, top_topics_filename)
    # Optional - save specified topic groups
    if 'topicGroups' in settings:
        topic_groups = settings['topicGroups']
        for topic_group in topic_groups:
            saveOverlappingPlot(settings, dft, topic_group, window_size, output_dir)
    print(f"Figures plotted to {output_dir}.")


def saveCoherencePlot(settings, coherence_values, topics_range, coherence_measure):
    output_dir = getOutputDir(settings)
    plt.plot(topics_range, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.savefig(os.path.join(output_dir, f'{settings["model"]}_{coherence_measure}_Coherence.png'))


### Interactive Web Page ###


def saveInteractivePage(settings, prepared_data):
    filename = f"InteractiveLDAvis.html"
    output_dir = getOutputDir(settings)
    path = os.path.join(output_dir, filename)
    pyLDAvis.save_html(prepared_data, path)
