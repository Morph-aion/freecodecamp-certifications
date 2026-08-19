# Énoncé freeCodeCamp — Mean-Variance-Standard Deviation Calculator

Source : https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/mean-variance-standard-deviation-calculator

## Instructions

Create a function named `calculate()` in `mean_var_std.py` that uses Numpy to output
the mean, variance, standard deviation, max, min, and sum of the rows, columns, and
elements in a 3 x 3 matrix.

The input of the function should be a list containing 9 digits. The function should
convert the list into a 3 x 3 Numpy array, and then return a dictionary containing
the mean, variance, standard deviation, max, min, and sum along both axes and for
the flattened matrix.

If the list contains fewer than 9 elements, raise a `ValueError` exception with the
message: `List must contain nine numbers.`

## Tests

The unit tests for this project are in `test_module.py`.

## Development

Write your code in `mean_var_std.py`. For development, you can use `main.py` to test
your `calculate()` function.

## Submitting

Copy your project's URL and submit it to freeCodeCamp.
