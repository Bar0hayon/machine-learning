#!/usr/bin/python3

from matplotlib import pyplot, pylab
import pandas
import numpy
from sklearn import linear_model

# read the data
df = pandas.read_csv("FuelConsumption.csv")

# take a look at the dataset
# print(df.head())

# selecting features for regression
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf.head(9))

# Lets plot Emission values with respect to Engine size:
# ==> REQUIRES GUI <==
pyplot.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
pyplot.xlabel("Engine size")
pyplot.ylabel("Emission")
# pyplot.show()

# train/test split
msk = numpy.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# gettin coefficients
regr = linear_model.LinearRegression()
x = numpy.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = numpy.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

# prediction
y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = numpy.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = numpy.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % numpy.mean((y_hat - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

# trying the model on the train dataset
print("prediction from the training data-set:")
print("--------------------------------------")
train_first_row = numpy.asanyarray([[train.values[0][0], train.values[0][1], train.values[0][4]]])
print("prediction: {}".format(regr.predict(train_first_row)[0][0]))
print("real: {}\n".format(train.values[0][5]))

# trying the model on the test dataset
print("prediction from the testing data-set:")
print("-------------------------------------")
test_first_row = numpy.asanyarray([[test.values[0][0], test.values[0][1], test.values[0][4]]])
print("prediction: {}".format(regr.predict(test_first_row)[0][0]))
print("real: {}".format(test.values[0][5]))
