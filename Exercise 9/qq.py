import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


students = pd.DataFrame(columns=['cmpt120', 'macm101', 'coop', 'admit', 'gpa'])


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer



# Normalize the colour data to 0 - 1 range
students = pd.DataFrame({'cmpt120':[120,33,44,55], 'macm101':[101,33,44,55], 'gpa':[3,4,3,2],'coop':['yes','yes','no','no'],'admit':['yes','yes','no','no']})

print(students)

def transform_data(students):
    scalar = StandardScaler()
    label = LabelEncoder()
    label.fit_transform(students[:,'coop'])
    label.fit_transform(students[:,'admit'])
    scalar.fit_transform(students['cmpt120', 'macm101', 'gpa'])
    return students

# Split the data
train, test = train_test_split(students, test_size=0.2)
# Training data
train_X = train[['cmpt120', 'macm101', 'coop', 'admit']]  # array with shape (n, 3). Divide by 255 so components are all 0-1.
train_y = train['gpa']  # array with shape (n,) of colour words.
# Validation data
test_X = test[['cmpt120', 'macm101', 'coop', 'admit']]
valid_y = test['gpa']

# Create naive Bayes classifer, train it
forest = RandomForestClassifier()
model = make_pipeline(
    FunctionTransformer(transform_data, validate=True),
    RandomForestClassifier())

model.fit(train_X.values, train_y.values)
# Test the model & Evaluate the accuracy score & print
score = model.score(test_X.values, valid_y.values)

