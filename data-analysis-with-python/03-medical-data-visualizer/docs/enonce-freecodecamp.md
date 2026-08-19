# Énoncé freeCodeCamp — Medical Data Visualizer

Source : https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/medical-data-visualizer

## Instructions

In this project, you will visualize and make calculations from medical examination
data using matplotlib, seaborn, and pandas. The dataset values were collected during
medical examinations.

### Data description

The rows in the dataset represent patients and the columns represent information
like body measurements, results from various blood tests, and lifestyle habits.
You will use the data to explore the relationship between cardiac disease, body
measurements, blood markers, and lifestyle choices.

File name: `medical_examination.csv`

### Tasks

Create a chart similar to `examples/Figure_1.png`, where we show the counts of good
and bad outcomes for the `cholesterol`, `gluc`, `alco`, `active`, and `smoke`
variables for patients with `cardio=1` and `cardio=0` in different panels.

Create a correlation matrix using the dataset. Plot the correlation matrix using
seaborn's `heatmap()`. Mask the upper triangle. The chart should look like
`examples/Figure_2.png`.

### Data cleaning

Filter the following points from the dataset:
- `ap_lo` should be less than or equal to `ap_hi`
- `height` should be less than or equal to the 97.5th percentile
- `height` should be greater than or equal to the 2.5th percentile
- `weight` should be less than or equal to the 97.5th percentile
- `weight` should be greater than or equal to the 2.5th percentile

### Feature engineering

Create a new column `overweight` in the dataset. Use the formula: `weight / height²`
(height in meters). If the value is greater than 25, the patient is overweight (1),
otherwise not (0).

Normalize the data by making 0 always good and 1 always bad. If the value of
`cholesterol` or `gluc` is 1, make the value 0. If the value is more than 1,
make the value 1.

## Tests

The unit tests for this project are in `test_module.py`.

## Development

Write your code in `medical_data_visualizer.py`. For development, you can use
`main.py` to test your functions.

## Submitting

Copy your project's URL and submit it to freeCodeCamp.
