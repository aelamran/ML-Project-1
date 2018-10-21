from scripts.proj1_helpers import *
from scripts.implementations import *

# Constants for importing training and testing data
TRAINING_DATA = '/home/ayyoubelamrani/Documents/ML/ML_course/projects/project1/all/train.csv'
TEST_DATA = '/home/ayyoubelamrani/Documents/ML/ML_course/projects/project1/all/test.csv'

# Load csv training and testing data
ty, tx, ids_train = load_csv_data(TRAINING_DATA, sub_sample=False)
#fy, fx, ids_test = load_csv_data(TEST_DATA, sub_sample=True)

#print(load_csv_data(TRAINING_DATA, sub_sample=True))

