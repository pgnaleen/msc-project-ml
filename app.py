from flask import Flask
# This wil use to split data into train and test, for model training
from sklearn.model_selection import train_test_split

# # SVM model
# from sklearn import svm

# # Stochastic Gradient Descent
# from sklearn.linear_model import SGDClassifier
# # feed data
# from sklearn.pipeline import make_pipeline

from prep_data import *
from models import *

warnings.filterwarnings('ignore')


app = Flask(__name__)


@app.route('/data-preprocessing/<file_name>')
def get_prep_data(file_name):
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


@app.route('/model_prep_train_test_evaluation')
def get_model_data():
    log = Logger()
    log.clean()
    log.start()
    model_preparation()
    log.stop()

    response = ""
    for ele in log.messages:
        response += ele
        response += "\n"

    print(log.messages)
    print(response)
    return response


def model_preparation():
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

    # making data split, x will contain the all column except our target column 'class', X have 17 columns
    x = df.drop(['class'], axis=1)
    # y will have only targeted column 'class', y have 1 column only
    y = df['class']

    # split the X and y into training and testing data,
    # training data will be used for model training and testing data will be used for model testing and predictions.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=50)

    # Print the training and testing dataset rows and columns size. first value is rows count, second is colum count
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # KNN model
    knn(x_train, y_train, x_test, y_test)

    # extreme gradient boosting
    xgb(x, y)

    # random forest classifier
    random_forest_classifier(x_train, y_train, x_test, y_test)

    # neural network
    neural_network(x_train, y_train, x_test, y_test, x, y)

    # naive bayes
    naive_bayes(x_train, y_train, x_test, y_test)


@app.route('/')
def anomaly_detection():
    return 'anomaly_detection!'


if __name__ == '__main__':
    app.run()
