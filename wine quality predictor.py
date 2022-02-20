import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error

#importing data from the csv file to a dataframe df using pandas library
df = pd.read_csv("winequality-red.csv")

#creating objects from the MinMxScalar for standardizing the data
mmc = MinMaxScaler()
x = df.iloc[:, 1:11].values
y = df.iloc[:, 11].values

#dividing the data into train set and testing set
train_x, test_x, train_y, test_y = model_selection.train_test_split(x, y, test_size=0.2)

train_x = mmc.fit_transform(train_x)
test_x = mmc.fit_transform(test_x)
svr = SVR(gamma="scale", kernel="rbf")

svr.fit(train_x, train_y)
pred_y = svr.predict(test_x)

#measuring the mean squared error and mean absolute error
print(mean_squared_error(test_y, pred_y))
print(mean_absolute_error(test_y, pred_y))