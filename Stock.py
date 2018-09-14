import pandas as pd
import numpy as np
import datetime
import quandl
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

# Get your dataframe of the stock
df = quandl.get("WIKI/AMZN")

type(df) # you can see you got a dataframe

print(df.tail(5))

# But we really only need the close column for predicting
df = df[['Adj. Close']]

# let's predict a month in advance, so make that var forecast out = frcst
# Make a new column for our label -- output. name it prediction
# simply shift it 30 units up
frcst = 180
df['Prediction'] = df[['Adj. Close']].shift(-frcst)

df.head()

# Make an array - X - with the Adj. Close values, but drop the X column
X = np.array(df.drop(['Prediction'], 1))
# normalize them
X = preprocessing.scale(X)

print(X)


X_frcst = X[-frcst:] # set frcst equal to last 30
# now we want to remove the last 30 from X
X = X[:-frcst] # remove last 30 from X

print(X)

# define output as y, and set it = to the Prediction column without the Nan data
y = np.array(df['Prediction'])
y = y[:-frcst]
print(y)

# Predict with linear regression
# We can set our test size to 20% of the total data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

# lets train now
classifier = LinearRegression()
classifier.fit(X_train, y_train)

confidence = classifier.score(X_test, y_test)
print('Confidence: ', confidence)

frcst_pred = classifier.predict(X_test)
print(frcst_pred)

plt.scatter(y_test, frcst_pred)

