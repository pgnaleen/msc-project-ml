# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# To load and process the csv files as dataframe
import pandas as pd
# to make numerical opertaion like, mapping or replacing values.
import numpy as np
# just to ignore the warnings.
import warnings
from flask import Flask
import sys
# to make graphs
import matplotlib.pyplot as plt
# To make numerical operations
import numpy as np
# This wil use to split data into train and test, for model training
from sklearn.model_selection import train_test_split
# check xgboost version
import xgboost
from xgboost import XGBClassifier
# Model evaluation, to measure the model accuracy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# Model evaluation metrics
from numpy import mean
from numpy import std
# KNN - model for prediction
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# Model evaluation
from sklearn.metrics import classification_report, confusion_matrix
# # Random Forest model
# from sklearn.ensemble import RandomForestClassifier
# # Keras Neural network model
# from keras.models import Sequential
# from keras.layers import Dense
# # SVM model
# from sklearn import svm
# # Naive Bayes model
# from sklearn.naive_bayes import GaussianNB
# # Stochastic Gradient Descent
# from sklearn.linear_model import SGDClassifier
# # feed data
# from sklearn.pipeline import make_pipeline

warnings.filterwarnings('ignore')


class Logger:
    stdout = sys.stdout
    messages = []

    def start(self):
        sys.stdout = self

    def stop(self):
        sys.stdout = self.stdout

    def write(self, text):
        self.messages.append(text)

    def clean(self):
        self.messages.clear()


app = Flask(__name__)


@app.route('/data-preprocessing/<file_name>')
def post_prep_data(file_name):
    # collect all console output logs
    log = Logger()
    log.clean()
    log.start()
    prep_data(file_name)
    log.stop()

    response = ""
    for ele in log.messages:
        response += ele
        response += "\n"

    print(log.messages)
    print(response)
    return response


@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


# This function is used to replace the text data with numerical values.
# Because model learns with number not with text, like making number a category is fit to model
# Like convert target column 'class' values from [suspicious,normal,unknown] to [0,1,2]

# text contains the text list, for example [suspicious,normal,unknown]
# numbs contains the number list, for example [0,1,2]
# columnName contains the column name , like class,Proto and Falgs, It iterate on columns one by one

# Note this function take much time, it requires many processing power,
# so you can also use already process dataset exorted in last of this notebook for the model file.
def replace_data(text, numbs, column_name, df=None):
    # Loop to iterate over one column on single time and update its values.
    for txt, num in zip(text, numbs):
        # Findig text as 'txt' and replacing it with number like 'num', updating the column as 'columnName'
        df[column_name] = np.where(df[column_name] == txt, num, df[column_name])


def prep_data(file_name):
    # Use a breakpoint in the code line below to debug your script.
    # Load the dataset into a dataframe
    # df = pd.read_csv("CIDDS-001-external-week1.csv")
    df = pd.read_csv(file_name)

    # Print the top 5 rows
    print(df.head())

    # # Class is our target column and our target column have 3 unique values, as printed.
    # df['class'].value_counts()
    #
    # # Drop these columns as data is missing in these columns.
    # df = df.drop(['attackType', 'attackID', 'attackDescription'], axis=1)
    #
    # # print the rest of the columns
    # print(df.columns)
    #
    # # Print if our dataset have any null or empty rows.
    # print(df.isnull().sum())
    #
    # # List of the columns we want to update from text to number
    # columns = ['class', 'Proto', 'Src IP Addr', 'Dst IP Addr', 'Bytes', 'Flags']
    #
    # # Calling function for each column, one by one
    # for col in columns:
    #     # Replace_data function takes two argument as described above.
    #     # 1 -  unique values on a column.
    #     # 2 -  Numbers list, so function can replace text with numbers
    #     # 3 -  Col is the column name, comes from the columns list
    #     replace_data(list(df[col].unique()), list(range(0, len(list(df[col].unique())))), col, df)
    #
    # # print the top 5 rows to see the changes
    # print(df.head())
    #
    # # We want to get the year, month, day, hour, minute as second as separate columns, so we will make date as datetime
    # # object Because
    # # to make a pattern of date transformation, like by each row there is a change in time, so model learn from this
    # # division of timeframe if we keep it as single colum, we have to make it a date index and it doesn't help in
    # # learning includes date time as well with this recognition
    # # it gives trends to data flow
    # df['Date first seen'] = pd.to_datetime(df['Date first seen'])
    #
    # df['Date first seen'] = pd.to_datetime(df['Date first seen'], format='%Y%m%d')
    #
    # # Here we are extratcing year,month and day as new columns, in each rows
    # df['year'] = pd.DatetimeIndex(df['Date first seen']).year
    # df['month'] = pd.DatetimeIndex(df['Date first seen']).month
    # df['day'] = pd.DatetimeIndex(df['Date first seen']).day
    # df['hour'] = pd.DatetimeIndex(df['Date first seen']).hour
    # df['min'] = pd.DatetimeIndex(df['Date first seen']).minute
    # df['sec'] = pd.DatetimeIndex(df['Date first seen']).second
    #
    # # Now we extract all columns as new, now we can drop the actual datetime column
    # # here inplace=True because......
    # # inplace true means update the dataframe with these changes
    # df.drop(['Date first seen'], axis=1, inplace=True)
    #
    # # print the top 5 rows to see the changes of datetime
    # print(df.head())
    #
    # # Save this process data as csv file, to be used for the models
    # df.to_csv("Processed_data.csv")


# a function to print the model results.
def conclude_model_results(test, pred):
    # confusion_matrix is a basic way to print the classifier model output.
    cm = confusion_matrix(test, pred)

    # Generating other accuracy parameters from the CM object of confusion matrix.

    tn = cm[0][0]
    print("True Negative")
    print(tn)
    fn = cm[1][0]
    print("False Negative")
    print(fn)
    tp = cm[1][1]
    print("True Positive")
    print(tp)
    fp = cm[0][1]
    print("False Positive")
    print(fp)

    precision = tp / (tp + fp)
    print("Precision")
    print(precision)

    recall = tp / (tp + fn)
    print("Recall")
    print(recall)

    # Overall accuracy
    acc = (tp + tn) / (tp + fp + fn + tn)
    print("Model Accuracy")
    print(acc)


# To plot the actual data and predicted data values, this is about the target column 'class'
def plot_test_pred(test, pred, y_test, y_pred1):
    # test have our test data, from the above data split.
    # pred have the list of prediction, sent by each model when we use it

    # make a dataframe, actual and predicted
    # why this y_pred1 gives error..........................
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})

    # plot the dataframe as barplot
    df1 = df.head(25)
    df1.plot(kind='bar', figsize=(16, 10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()


# show the output of the model
def model_summary(test, pred):
    print(confusion_matrix(test, pred))
    # what this classification_report does............................................
    print(classification_report(test, pred))


def model_preparation(df):
    # Load the process data, saved by data_prep file
    df = pd.read_csv("Processed_data.csv")

    # avoid load the unnamed columns
    # why avoid unnamed columns?
    # when we save data, it adds one column as index with no name, it's redundant so just avoiding dropping and not
    # loading
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # print the rows
    df.head()

    # convert float data into int, to remove the values decimals points
    df.Duration = df.Duration.astype(int)
    df['Dst Pt'] = df['Dst Pt'].astype(int)
    df.head()

    # Unique values in class column. suspicious,normal and unknown replaced with 0,1,2
    df['class'].value_counts()

    # Check the null values
    df.isnull().sum()

    df.head()

    df['class'].unique()

    # making data split, X will contain the all column except our target column 'class', X have 17 columns
    X = df.drop(['class'], axis=1)
    # y will have only target column 'class', y have 1 column only
    y = df['class']

    # split the X and y into training and testing data,
    # training data will be used for model traiing and testing data will be used for model testing and predictions.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50)

    # Print the training and testing dataset rows and columns size. first value is rows count, second is colum count
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    ################KNN###################
    # make a object
    scaler = StandardScaler()
    # fit the object to training data
    scaler.fit(X_train)

    # make dataset values normalize
    X_train1 = scaler.transform(X_train)
    X_test1 = scaler.transform(X_test)

    # assign the data
    y_train1 = y_train
    y_test1 = y_test

    # make object of knn model
    classifier = KNeighborsClassifier(n_neighbors=5)

    # train the knn model on training datset
    classifier.fit(X_train1, y_train1)

    # Make predictions on the test dataset
    y_pred1 = classifier.predict(X_test1)

    # print the result, just pass the prediction from above tab, and test data for both functions.
    print(confusion_matrix(y_test1, y_pred1))
    print(classification_report(y_test1, y_pred1))

    # pass test and predicted list to print the results.
    conclude_model_results(y_test, y_pred1)

    # pass test and predicted list to print the model summary.
    model_summary(y_test, y_pred1)

    # pass test and predicted list to print the model performance in shape or barplot.
    plot_test_pred(y_test, y_pred1)


if __name__ == '__main__':
    app.run()
