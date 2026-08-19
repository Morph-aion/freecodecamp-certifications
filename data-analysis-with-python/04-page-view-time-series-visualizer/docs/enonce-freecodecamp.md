# Énoncé freeCodeCamp — Page View Time Series Visualizer

Source : https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/page-view-time-series-visualizer

## Instructions

For this project you will visualize time series data using a line chart, a bar chart,
and box plots. You will use [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/),
and [Seaborn](https://seaborn.pydata.org/) to visualize a dataset containing the number
of page views each day on the freeCodeCamp forum from 2016-05-09 to 2019-12-31.

### Data cleaning

Use the data if you need to clean it. You should filter the day-ticks to show only
the data from May 2016 to December 2019. Remove the 2.5% of the lowest values and
the 2.5% of the highest values.

### Tasks

Create a `draw_line_plot` function that uses Matplotlib to draw a line chart similar
to `examples/Figure_1.png`. The title should be
`Daily freeCodeCamp Forum Page Views 5/2016-12/2019`. The label on the x-axis should
be `Date` and the label on the y-axis should be `Page Views`.

Create a `draw_bar_plot` function that draws a bar chart similar to
`examples/Figure_2.png`. It should show the average daily page views for each month
grouped by year. The legend should show month labels and have a title of `Months`.
On the chart, the label on the x-axis should be `Years` and the label on the y-axis
should be `Average Page Views`.

Create a `draw_box_plot` function that uses Seaborn to draw two adjacent box plots
similar to `examples/Figure_3.png`. These box plots should show how the values are
distributed within a given year or month and how it compares over time. The title of
the first chart should be `Year-wise Box Plot (Trend)` and the title of the second
chart should be `Month-wise Box Plot (Seasonality)`. Make sure the month labels on the
bottom start at `Jan` and the x and y axis are labeled correctly. The boilerplate
includes commands to prepare the data.

## Tests

The unit tests for this project are in `test_module.py`.

## Development

Write your code in `time_series_visualizer.py`. For development, you can use
`main.py` to test your functions.

## Submitting

Copy your project's URL and submit it to freeCodeCamp.
