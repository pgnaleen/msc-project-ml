import tensorflow as tensorflow
# !pip install tensorflow
from utils import *
from sklearn.metrics import classification_report, confusion_matrix

# KNN - model for prediction
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# check xgboost version
from xgboost import XGBClassifier
# Model evaluation, to measure the model accuracy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
# Model evaluation metrics
from numpy import mean
from numpy import std

# Random Forest model
from sklearn.ensemble import RandomForestClassifier

# Keras Neural network model
from keras.models import Sequential
from keras.layers import Dense

# Naive Bayes model
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')


def knn(x_train, y_train, x_test, y_test):
    print('knn model processing started...')
    # make a object
    scaler = StandardScaler()
    # fit the object to training data
    scaler.fit(x_train)

    # make dataset values normalize
    x_train1 = scaler.transform(x_train)
    x_test1 = scaler.transform(x_test)

    # assign the data
    y_train1 = y_train
    y_test1 = y_test

    # make object of knn model
    classifier = KNeighborsClassifier(n_neighbors=5)

    # train the knn model on training dataset
    classifier.fit(x_train1, y_train1)

    # Make predictions on the test dataset
    y_pred1 = classifier.predict(x_test1)

    # print the result, just pass the prediction from above tab, and test data for both functions.
    print(confusion_matrix(y_test1, y_pred1))
    print(classification_report(y_test1, y_pred1))

    # pass test and predicted list to print the results.
    conclude_model_results(y_test, y_pred1)

    # pass test and predicted list to print the model summary.
    model_summary(y_test, y_pred1)

    # pass test and predicted list to print the model performance in shape or bar plot.
    plot_test_pred(y_test, y_pred1)


def xgb(x, y):
    print('xgb model processing started...')
    # define the model
    model = XGBClassifier()

    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)

    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


def random_forest_classifier(x_train, y_train, x_test, y_test):
    print('random_forest_classifier model processing started...')
    # make object of rfc model
    clf = RandomForestClassifier(max_depth=2, random_state=0)

    # train the rfc model on training datset
    clf.fit(x_train, y_train)

    # Make predictions on the test dataset
    pred2 = clf.predict(x_test)

    # print the result, just pass the prediction from above tab, and test data for both functions.
    print(confusion_matrix(y_test, pred2))
    print(classification_report(y_test, pred2))


def neural_network(x_train, y_train, x_test, y_test, x, y):
    print('neural_network model processing started...')
    # this is model object
    model = Sequential()

    # this is input layer, have 17 inputs, with relu as activation function
    model.add(Dense(17, input_dim=17, activation='relu'))

    # these are hidden layers with differn nodes count and with relu as activation function
    model.add(Dense(34, activation='relu'))
    model.add(Dense(34, activation='relu'))
    model.add(Dense(34, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(6, activation='relu'))

    # this is output layer with 1 node and sigmoid as activation function
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model, with adam as optimizer and binary_crossentropy as loss function.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset, run model for 10 times, with batch size of 20000
    model.fit(x_train, y_train, epochs=10, batch_size=20000)

    # Make predictions on the test dataset
    y_pred1 = model.predict(x_test)

    # evaluate the keras model
    _, accuracy = model.evaluate(x, y)
    print('Accuracy: %.2f' % (accuracy * 100))


def naive_bayes(x_train, y_train, x_test, y_test):
    print('naive_bayes model processing started...')
    # make a object
    gnb = GaussianNB()

    # train the knn model on training datset
    gnb.fit(x_train, y_train)

    # Make predictions on the test dataset
    y_pred = gnb.predict(x_test)

    # print the result, just pass the prediction from above tab, and test data for both functions.
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
