# go-ml

This is a simple command line utility for running basic machine learning training and predictions written in go.

Currently only linear regression is implemented using gradient descent. However, more features are planned.

## Features

The tool will accept input from a CSV file, train a model and output trained model parameters as well as test data for the trained models.

The provided input file will be shuffled into a random order and then chunked into a training set, cross validation set and test set. The data will then be mean normalized and the training set fed into the gradient descent algorithm.

Once trained, the model will be tested against the training set, cross validation set and test sets and the percentage error will be outputted.

The test set and a results file will also be outputted where they can be tested independently and the expected vs. predicted can be outputted.

## Usage

```bash
$ linear_regression -h
Usage of linear_regression:
  -a float
        Learning Rate (alpha) (default 0.5)
  -i int
        # of iterations of gradient descent for learning (default 5000)
  -in string
        Input CSV file (default "./input.csv")
  -l float
        Regularization Constant (lambda) (default 0.01)
  -out string
        Output directory (default "./")
```

```bash
$ linear_regression -in ./test_data/input.csv -out ./test_data -a 0.5 -l 0.01 -i 5000
Training Started.
+----------------------+-------------------+--------------------+
| TRAINING ERROR ('%') |    CV ERROR ('%') |   TEST ERROR ('%') |
+----------------------+-------------------+--------------------+
|   2.9188262471947364 | 2.870388389393404 | 3.2847186730766893 |
+----------------------+-------------------+--------------------+
Training Done.
Duration: 189.503201ms
```

### Input File

The input file expects a format as follows. Where each column in the CSV file will be interpreted as a feature of the model and the last column will be expected to be the output for the training data.

NOTE: The file is expected to represent x1, x2...xn. x0, where x0 = 1 for all features will automatically be added to the features, this does not need to be added manually in the input file.

```csv
3,1.732050808,0.5705446405
3.01,1.734935157,0.5740016588
3.02,1.73781472,0.5574277229
3.03,1.740689519,0.5477046603
3.04,1.743559577,0.5053156969
3.05,1.74642492,0.5395282651
3.06,1.749285568,0.5814740561
3.07,1.752141547,0.5096136798
3.08,1.754992877,0.5562753539
3.09,1.757839583,0.5444135702
...
..
.
```

### Output Files

The tool outputs two files after running successfully, these are a JSON file containing all of the parameters of the analysis and a CSV file containing a test set of data including the features, actual outputs and predicted outputs.

#### Test Data Set (output.csv)

The format of the output file is as follows: x0, x1...xn, y, y_pred, where y is the actual output from the test set of the training data and y_pred is the predicted output from the model.

```csv
1.000000,-0.481611,-0.511687,0.552152,0.586945
1.000000,0.414654,0.374601,1.099460,1.125203
1.000000,0.048472,0.072659,1.001968,0.958102
1.000000,0.258015,0.251698,1.085337,1.059216
1.000000,0.357600,0.330772,1.091743,1.101991
1.000000,0.292248,0.279269,1.138151,1.074261
1.000000,0.069219,0.091265,0.944965,0.968885
1.000000,0.408430,0.369868,1.097626,1.122714
1.000000,0.337891,0.315390,1.115979,1.093761
1.000000,0.207185,0.209956,1.025387,1.036170
.....
....
...
..
.
```

#### Results (results.json)

Theta values are listed in order Theta0, Theta1...ThetaN.

```json
{
  "Thetas": [0.9072869372479312, -0.26747590992580883, 0.8778052636513626],
  "Alpha": 0.5,
  "Lambda": 0.01,
  "NumIterations": 5000,
  "TrainingError": 2.9188262471947364,
  "CVError": 2.870388389393404,
  "TestError": 3.2847186730766893
}
```

## Sample Data

The sample data provided in this repo under `test_data/input.csv` is a basic data set where the output is a segment of the output of the `log` function with some random noise applied to it. The features in the input file represent the following formula `a*1 + b*x + c*sqrt(x)`. This happens to give a pretty descent approximation of the `log` function for a certain range.

## TODO

- Implement predictions based on a provided model file
- Implement performance graphs (learning curves)
- Implement result visualisation
- Implement hyperparameter tuning
- Implement logistic regression
