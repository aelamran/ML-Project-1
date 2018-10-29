# ML-Project-1 Machine learning EPFL
This machine learning project purpose is the prediction of the boson higgs based on a large dataset using only numpy and python standard libraries.

## Kaggle Team Name:
EL PREDICTOR

## Team members:
Ayyoub EL AMRANI (ayyoub.elamrani@epfl.ch)
Fatine BENHSAIN (fatine.benhsain@epfl.ch)

## Final Kaggle Accuracy: 
0.82924

## Important:
In order to run the code, please indicate the location of the training data and testing data in the run.py script. The variables to modify are specified as TRAINING_DATA and TEST_DATA.

* __run.py:__ This script runs our implentation for training the model and generating the labels for the testing dataset. In order to run it, go to the concerned directory and enter 'python run.py' on your terminal. Make sure you have the same version of Python (3.6) otherwise some results may vary. A new file 'output.csv' should be created in the same directory containing the labels that we have submitted on the Kaggle page.

* __implementations.py:__ This script contains all the needed methods in order to run our algorithm. Indeed, it contains the the methods that we were asked to implement in the descriptif of the project and other methods that we needed to use in our process.

* __proj1_helpers.py:__ original helpers provided to the class that allow us to load input csv files, generate predictions from weights, and create a csv submission.