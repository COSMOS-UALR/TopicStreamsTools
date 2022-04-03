# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Add
- Seamless node switching

## [0.10.0] - 2022-02-04
### Added
- Commenter Network Node

## [0.9.4] - 2022-02-07
### Changed
- Date ticks display for Plotly. Some useful resources for further customization are [this guide](https://plotly.com/python/time-series/#configuring-tick-labels) and [the documentation](https://plotly.com/python/reference/layout/xaxis/). See the `setPlotlyOptions` method to customize.

## [0.9.3] - 2022-02-04
### Changed
- Improved plot options (logic & better axis labels)

## [0.9.2] - 2022-02-04
### Added
- Interactive plotly output for multiple topics.
- `alternate_legend` setting on `True` will show legends on the right of the plots.

## [0.9.1] - 2021-11-23
### Added
- Multithreading for model optimization.

### Known issues
- Using multithreading will cause an error on plot generation. Rerunning the script without retraining (loading the optimized model) will be successful.

## [0.9.0] - 2021-11-02
### Added
- Accepts stopwords from `stopwords.txt`. This filters out tokens at corpus processing.
- The following parameters can now be informed in the config file when using the Gensim LDA model (see [the documenation](https://radimrehurek.com/gensim/models/ldamodel.html) for details):
    - passes
    - iterations
- The following parameters can now be informed in the config file when using the LDA Mallet model (see [the documenation](https://radimrehurek.com/gensim_3.8.3/models/wrappers/ldamallet.html) for details):
    - iterations

### Changed
- The default LDA alpha is now set to [auto](https://radimrehurek.com/gensim/models/ldamodel.html).

<!-- ### Removed
- Obsolete conf file -->
