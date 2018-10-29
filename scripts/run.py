from scripts.implementations import *

# Constants for importing training and testing data
TRAINING_DATA = '/home/ayyoubelamrani/Documents/ML/ML_course/projects/project1/all/train.csv'
TEST_DATA = '/home/ayyoubelamrani/Documents/ML/ML_course/projects/project1/all/test.csv'

# Ridge regression parameters
SPLIT_RATIO = 0.80
DEGREE = 11
LAMBDA = 0.8357142857142857

print("Loading data from train = " + TRAINING_DATA + " test = " + TEST_DATA)
print("------------------LOADING DATA------------------")

# Load csv training and testing data
training_y, training_x, ids_train = load_csv_data(TRAINING_DATA, sub_sample=False)
submission_y, submission_x, ids_test = load_csv_data(TEST_DATA, sub_sample=False)

print("------------------DATA LOADED------------------.")

# Backup of the imported training and testing data
training_x_backup = training_x.copy()
training_y_backup = training_y.copy()
submission_x_backup = submission_x.copy()

print("------------------PRECESSING------------------")

# Split data into 80-20 and use 80 for training and 20 to check model accuracy
x_train, y_train, x_test, y_test = split_data(training_x_backup.copy(), training_y_backup.copy(), SPLIT_RATIO, seed=1)

# Split final test data to make predictions on into buckets
submission_x_train = submission_x_backup.copy()
submission_x_buckets = get_buckets(submission_x_train)

# Append y values as column to later divide y into buckets corresponding with x values
x_train = np.column_stack((x_train, y_train))
x_test = np.column_stack((x_test, y_test))

# Split training x into buckets
buckets = get_buckets(x_train)

# Split testing y into buckets corresponding to x values
y_buckets = []
for i in range(len(buckets)):
    y_buckets.append(buckets[i][:, -1])
    buckets[i] = np.delete(buckets[i], -1, 1)

# Split testing x into buckets
test_buckets = get_buckets(x_test)

# Split testing y into buckets corresponding to x values
test_y_buckets = []
for i in range(len(test_buckets)):
    test_y_buckets.append(test_buckets[i][:, -1])
    test_buckets[i] = np.delete(test_buckets[i], -1, 1)

# Standardize columns for each bucket
for b in range(len(buckets)):
    buckets[b] = normalize(buckets[b])
    test_buckets[b] = normalize(test_buckets[b])
    submission_x_buckets[b] = normalize(submission_x_buckets[b])

# Build polynomial of given degree for each bucket
for b in range(len(buckets)):
    buckets[b] = build_poly(buckets[b], DEGREE)
    test_buckets[b] = build_poly(test_buckets[b], DEGREE)
    submission_x_buckets[b] = build_poly(submission_x_buckets[b], DEGREE)

# Add column of ones for intercept for each bucket
for b in range(len(buckets)):
    buckets[b] = np.column_stack((np.ones((buckets[b].shape[0], 1)), buckets[b]))
    test_buckets[b] = np.column_stack((np.ones((test_buckets[b].shape[0], 1)), test_buckets[b]))
    submission_x_buckets[b] = np.column_stack((np.ones((submission_x_buckets[b].shape[0], 1)), submission_x_buckets[b]))

print("------------------TRAINING OF RIDGE REGRESSION------------------")

# Calculate weights for each bucket separately
weights = []
for i in range(len(buckets)):
    w_rr, loss_rr = ridge_regression(y_buckets[i], buckets[i], LAMBDA)
    weights.append(w_rr)

# Compare predictions for each bucket using its corresponding weights found earlier
correct_predictions = 0
len_data = 0
for i in range(len(buckets)):
    rr_accuracy = compute_accuracy(weights[i], test_buckets[i], test_y_buckets[i])
    correct_predictions += (rr_accuracy * len(test_buckets[i]))
    len_data += len(test_buckets[i])

print("------------------TRAINING COMPLETED------------------")
total_accuracy = correct_predictions / len_data
print("Total Accuracy = " + str(total_accuracy) + " Degree = " + str(DEGREE) + " Lambda = " + str(LAMBDA))


# Create new array with Id, PRI_jet_num, and DER_mass_MMC for reordering predictions
ids_array = ids_test
pri_jet_num_feature = submission_x_backup[:, 22]
der_mass_mmc_col_feature = submission_x_backup[:, 0]
ids_array = np.column_stack((ids_array, pri_jet_num_feature))
ids_array = np.column_stack((ids_array, der_mass_mmc_col_feature))

# Divide Id into 8 buckets similar to input data
id_buckets = get_id_buckets(ids_array)

# Make predictions for each bucket using weights calculated by training on each bucket
submission = predict_labels(weights[0], submission_x_buckets[0])
submission = np.column_stack((submission, id_buckets[0]))
for i in range(1, len(weights)):
    predictions = predict_labels(weights[i], submission_x_buckets[i])
    predictions = np.column_stack((predictions, id_buckets[i]))
    submission = np.concatenate((submission, predictions))

# Sort predictions based on Id
submission = submission[submission[:, 1].argsort()]

# Select only prediction values
submission = submission[:, 0]

# Create output file containing predictions
create_csv_submission(ids_test, submission, "output.csv")
print("Created output.csv with shape = " + str(submission.shape))