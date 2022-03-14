# just to ignore the warnings.
import warnings
import sys
# To load and process the csv files as dataframe
import pandas as pd
# to make graphs
# to make numerical operation like, mapping or replacing values.
import numpy as np
import matplotlib.pyplot as plt
# Model evaluation
from sklearn.metrics import classification_report, confusion_matrix

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

    def flush(self):
        pass


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


# show the output of the model
def model_summary(test, pred):
    print(confusion_matrix(test, pred))
    # what this classification_report does............................................
    print(classification_report(test, pred))


# To plot the actual data and predicted data values, this is about the target column 'class'
def plot_test_pred(y_test, y_pred1):
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
