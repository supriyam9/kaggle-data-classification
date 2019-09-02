# This was my submission for a Kaggle Data Classification Competition.
# The below algorithm predicts whether a customer will make a payment
# based on their previous payment history and additional details.
# Please see the train.csv for training data and the submission.csv to
# see the output of the binary classification model.



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import os


# Read the CSV for training data
train_data = pd.read_csv("../input/train.csv")
train_data = training_data.dropna()

# Replace any invalid/implausible entries with placeholder entry
df['PAST_1'][df['PAST_1'] == -2] = -1
df['PAST_2'][df['PAST_2'] == -2] = -1
df['PAST_3'][df['PAST_3'] == -2] = -1
df['PAST_4'][df['PAST_4'] == -2] = -1
df['PAST_5'][df['PAST_5'] == -2] = -1
df['PAST_6'][df['PAST_6'] == -2] = -1


#Partition the dataset
print(df.head())
X_2 = training_data.drop('PAID_NEXT_MONTH', axis=1)
y = training_data['PAID_NEXT_MONTH']

# Scale the training data to help the model learn quickly
X = scale(X_2)

# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# Instantiate a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=90)
clf.fit(X_train, y_train)

# Use the model to predict results and get the accuracy
y_pred = clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test, y_pred))


# Get the test submission file and check prediction values
testDf = pd.read_csv("../input/test.csv")
testDf = testDf.drop('PAID_NEXT_MONTH', axis=1)
testDf.dropna()
results=clf.predict(testDf)
idNum = testDf['ID']

# Format & write submission file with ID and customer payment prediction
submission = pd.DataFrame({'Id': idNum, 'PAID_NEXT_MONTH': results})
submission.to_csv('submission.csv', index=False)


# Upon submitting the final predictions to the competition, this algorithm
# had approximately 82% accuracy.
